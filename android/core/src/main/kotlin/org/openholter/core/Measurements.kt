package org.openholter.core

import java.io.File
import java.time.ZoneId

/** A time measurement between two ECG samples (Unix ns), drawn by the owner on a marker report. */
data class Measurement(val aNs: Long, val bNs: Long) {
    val intervalMs: Double get() = Math.abs(bNs - aNs) / 1e6

    /** The rate an interval of this length corresponds to (60 000 / ms). */
    val bpm: Double get() = if (intervalMs > 0) 60_000.0 / intervalMs else 0.0
}

/**
 * Measurements per marker, one small CSV per day of the marker
 * (`measurements_<day>.csv`, columns [RecordingFiles.MEASUREMENTS_HEADER];
 * `interval_ms` is for reading the file, the times are the data).
 */
class MeasurementStore(private val root: File, private val zone: ZoneId) {

    @Synchronized
    fun list(markerNs: Long): List<Measurement> =
        readAll(markerNs).filter { it.first == markerNs }.map { it.second }

    @Synchronized
    fun add(markerNs: Long, m: Measurement) = write(markerNs, readAll(markerNs) + (markerNs to m))

    @Synchronized
    fun remove(markerNs: Long, m: Measurement) = write(markerNs, readAll(markerNs).filterNot { it.first == markerNs && it.second == m })

    private fun file(markerNs: Long) = RecordingFiles.measurementsFile(root, RecordingFiles.dayOf(markerNs, zone))

    private fun readAll(markerNs: Long): List<Pair<Long, Measurement>> {
        val f = file(markerNs)
        if (!f.exists()) return emptyList()
        return f.readLines().mapNotNull { line ->
            val c = line.split(',')
            val marker = c.getOrNull(0)?.toLongOrNull() ?: return@mapNotNull null // header
            val a = c.getOrNull(1)?.toLongOrNull() ?: return@mapNotNull null
            val b = c.getOrNull(2)?.toLongOrNull() ?: return@mapNotNull null
            marker to Measurement(a, b)
        }
    }

    private fun write(markerNs: Long, rows: List<Pair<Long, Measurement>>) {
        val f = file(markerNs)
        f.parentFile?.mkdirs()
        val tmp = File(f.parentFile, f.name + ".tmp")
        tmp.writeText(
            buildString {
                append(RecordingFiles.MEASUREMENTS_HEADER).append('\n')
                for ((marker, m) in rows.sortedWith(compareBy({ it.first }, { minOf(it.second.aNs, it.second.bNs) }))) {
                    append(marker).append(',').append(m.aNs).append(',').append(m.bNs).append(',')
                    append("%.1f".format(java.util.Locale.ROOT, m.intervalMs)).append('\n')
                }
            },
        )
        if (!tmp.renameTo(f)) {
            f.delete()
            tmp.renameTo(f)
        }
    }
}

/** Where a tapped point lands on the ECG. */
object EcgSnap {
    /** Default search half-width for [peak]: enough to reach the tip of a beat near the tap. */
    const val PEAK_RADIUS_NS = 50_000_000L

    /**
     * Time of the highest sample within [radiusNs] of [tNs] (the tip of the beat near a tap), or
     * of the nearest sample when there is none in that range; null for an empty window.
     */
    fun peak(w: EcgWindow, tNs: Long, radiusNs: Long = PEAK_RADIUS_NS): Long? {
        if (w.size == 0) return null
        val i0 = w.t.binarySearch(tNs - radiusNs).let { if (it < 0) -it - 1 else it }
        val i1 = w.t.binarySearch(tNs + radiusNs + 1).let { if (it < 0) -it - 1 else it }
        if (i1 <= i0) return nearest(w, tNs)
        var best = i0
        for (i in i0 until i1) if (w.mv[i] > w.mv[best]) best = i
        return w.t[best]
    }

    /** Time of the sample nearest to [tNs]; null for an empty window. */
    fun nearest(w: EcgWindow, tNs: Long): Long? {
        if (w.size == 0) return null
        val i = w.t.binarySearch(tNs).let { if (it < 0) -it - 1 else it }
        return when {
            i <= 0 -> w.t[0]
            i >= w.size -> w.t[w.size - 1]
            tNs - w.t[i - 1] <= w.t[i] - tNs -> w.t[i - 1]
            else -> w.t[i]
        }
    }

    /** The ECG value (mV) at the sample nearest to [tNs], or null. */
    fun valueAt(w: EcgWindow, tNs: Long): Float? {
        val t = nearest(w, tNs) ?: return null
        val i = w.t.binarySearch(t)
        return if (i >= 0) w.mv[i] else null
    }
}
