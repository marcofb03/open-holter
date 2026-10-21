package org.openholter.core

import java.io.Closeable
import java.io.File
import java.time.ZoneId

/**
 * Writes ECG in the ECGLogger CSV layout:
 *
 * ```
 * timestamp_ns,ecg_mV,pulse_bpm,rr_ms,marker
 * 1729512000000000000,0.123,,,
 * 1729512000007692307,0.131,62,968,
 * 1729512000015384615,0.140,,,1
 * ```
 *
 * One row per ECG sample. `pulse_bpm`, `rr_ms` and `marker` are only filled
 * when there is something new: each heart-rate value, RR interval and marker
 * is attached to the first ECG row whose timestamp is at or after the event.
 * If no ECG row arrives within [staleNs] (e.g. the strap is disconnected),
 * the event is written as its own row with an empty `ecg_mV`; `ecgpipe`
 * accepts such rows.
 *
 * Not thread-safe: call from one thread.
 */
class EcgCsvWriter(
    root: File,
    zone: ZoneId,
    flushIntervalNs: Long = 10_000_000_000L,
    syncIntervalNs: Long = 60_000_000_000L,
    private val staleNs: Long = 3_000_000_000L,
    private val gapThresholdNs: Long = 1_000_000_000L,
    /** Called when two consecutive ECG samples are more than [gapThresholdNs] apart. */
    private val onGap: (fromNs: Long, toNs: Long) -> Unit = { _, _ -> },
) : Closeable {

    private val file = HourlyFileWriter(
        zone, RecordingFiles.ECG_HEADER, { RecordingFiles.ecgFile(root, it) },
        flushIntervalNs, syncIntervalNs,
    )
    private val pulses = ArrayDeque<TimedValue>()
    private val rrs = ArrayDeque<TimedValue>()
    private val markers = ArrayDeque<Long>()
    private val row = StringBuilder(64)

    var lastEcgNs: Long = Long.MIN_VALUE
        private set

    val currentFile: File? get() = file.currentFile

    fun writeEcg(timestampNs: Long, microvolts: Int) {
        if (lastEcgNs != Long.MIN_VALUE && timestampNs - lastEcgNs > gapThresholdNs) {
            onGap(lastEcgNs, timestampNs)
        }
        lastEcgNs = timestampNs
        row.setLength(0)
        row.append(timestampNs).append(',')
        appendMillivolts(row, microvolts)
        row.append(',')
        pulses.takeDue(timestampNs)?.let { row.append(it) }
        row.append(',')
        rrs.takeDue(timestampNs)?.let { row.append(it) }
        row.append(',')
        if (markers.isNotEmpty() && markers.first() <= timestampNs) {
            // Several presses before one sample collapse into one marked row.
            while (markers.isNotEmpty() && markers.first() <= timestampNs) markers.removeFirst()
            row.append('1')
        }
        file.writeLine(timestampNs, row)
    }

    fun addHeartRate(timestampNs: Long, bpm: Int) {
        if (bpm > 0) pulses.addLast(TimedValue(timestampNs, bpm))
    }

    fun addRr(timestampNs: Long, rrMs: Int) {
        if (rrMs > 0) rrs.addLast(TimedValue(timestampNs, rrMs))
    }

    fun addMarker(timestampNs: Long) {
        markers.addLast(timestampNs)
    }

    /** Call regularly (about once a second) with the current wall-clock time. */
    fun tick(nowNs: Long) {
        writeStandaloneRows(nowNs - staleNs)
        file.tick(nowNs)
    }

    fun flush(sync: Boolean = false) = file.flush(sync)

    /** Writes everything still pending and closes the current file. */
    override fun close() {
        writeStandaloneRows(Long.MAX_VALUE)
        file.close()
    }

    /** Writes pending events with timestamp <= [upToNs] as rows without ECG. */
    private fun writeStandaloneRows(upToNs: Long) {
        while (true) {
            val t = minOf(
                pulses.firstOrNull()?.timestampNs ?: Long.MAX_VALUE,
                rrs.firstOrNull()?.timestampNs ?: Long.MAX_VALUE,
                markers.firstOrNull() ?: Long.MAX_VALUE,
            )
            if (t == Long.MAX_VALUE || t > upToNs) return
            row.setLength(0)
            row.append(t).append(",,")
            if (pulses.firstOrNull()?.timestampNs == t) row.append(pulses.removeFirst().value)
            row.append(',')
            if (rrs.firstOrNull()?.timestampNs == t) row.append(rrs.removeFirst().value)
            row.append(',')
            if (markers.firstOrNull() == t) {
                markers.removeFirst()
                row.append('1')
            }
            file.writeLine(t, row)
        }
    }

    private fun ArrayDeque<TimedValue>.takeDue(timestampNs: Long): Int? =
        if (isNotEmpty() && first().timestampNs <= timestampNs) removeFirst().value else null

    private data class TimedValue(val timestampNs: Long, val value: Int)

    companion object {
        /** Exact µV → mV conversion without floating point, e.g. -1234 → "-1.234". */
        fun appendMillivolts(sb: StringBuilder, microvolts: Int) {
            val abs = Math.abs(microvolts.toLong())
            if (microvolts < 0) sb.append('-')
            sb.append(abs / 1000).append('.')
            val frac = abs % 1000
            if (frac < 100) sb.append('0')
            if (frac < 10) sb.append('0')
            sb.append(frac)
        }

        fun formatMillivolts(microvolts: Int): String =
            StringBuilder(12).also { appendMillivolts(it, microvolts) }.toString()
    }
}
