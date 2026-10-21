package org.openholter.core

import java.io.Closeable
import java.io.File
import java.time.ZoneId

/**
 * The H10's own heart-rate notifications (about 1 per second), hourly files:
 *
 * ```
 * timestamp_ns,hr_bpm,rr_ms,contact,contact_supported
 * 1729512000512000000,62,968,1,1
 * 1729512001498000000,63,,1,1
 * ```
 *
 * One row per RR interval in the notification (rows of one notification share
 * its timestamp, the phone time it arrived); a notification without RR gives
 * one row with an empty `rr_ms`. `contact` / `contact_supported` are 1 or 0.
 */
class HrCsvWriter(
    root: File,
    zone: ZoneId,
    flushIntervalNs: Long = 10_000_000_000L,
    syncIntervalNs: Long = 60_000_000_000L,
) : Closeable {

    private val file = HourlyFileWriter(
        zone, RecordingFiles.HR_HEADER, { RecordingFiles.hrFile(root, it) },
        flushIntervalNs, syncIntervalNs,
    )
    private val row = StringBuilder(48)

    fun write(timestampNs: Long, bpm: Int, rrsMs: List<Int>, contact: Boolean, contactSupported: Boolean) {
        if (rrsMs.isEmpty()) {
            writeRow(timestampNs, bpm, null, contact, contactSupported)
        } else {
            for (rr in rrsMs) writeRow(timestampNs, bpm, rr, contact, contactSupported)
        }
    }

    fun tick(nowNs: Long) = file.tick(nowNs)

    fun flush(sync: Boolean = false) = file.flush(sync)

    override fun close() = file.close()

    private fun writeRow(timestampNs: Long, bpm: Int, rr: Int?, contact: Boolean, supported: Boolean) {
        row.setLength(0)
        row.append(timestampNs).append(',').append(bpm).append(',')
        if (rr != null) row.append(rr)
        row.append(',').append(if (contact) '1' else '0')
        row.append(',').append(if (supported) '1' else '0')
        file.writeLine(timestampNs, row)
    }
}
