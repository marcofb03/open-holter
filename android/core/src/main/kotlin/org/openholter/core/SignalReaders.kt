package org.openholter.core

import java.io.File
import java.time.ZoneId

/** ECG samples in a time window: phone time (Unix ns) and millivolts, rows without ECG left out. */
class EcgWindow(val t: LongArray, val mv: FloatArray) {
    val size: Int get() = t.size
}

/** One bin of the H10's heart rate: median, min and max of the readings in it. */
data class HrBin(val startNs: Long, val median: Int, val min: Int, val max: Int, val count: Int)

/** Reads parts of the hourly files back for the app's charts and marker reports. */
object SignalReaders {

    /** ECG samples with [fromNs] <= t < [toNs] from the hourly `ecg_*.csv` files. */
    fun ecg(root: File, zone: ZoneId, fromNs: Long, toNs: Long): EcgWindow {
        var t = LongArray(1024)
        var v = FloatArray(1024)
        var n = 0
        for (file in hourlyFiles(fromNs, toNs, zone) { RecordingFiles.ecgFile(root, it) }) {
            file.bufferedReader().useLines { lines ->
                for (line in lines) {
                    val a = line.indexOf(',')
                    if (a <= 0) continue
                    val ts = line.substring(0, a).toLongOrNull() ?: continue // header
                    if (ts < fromNs || ts >= toNs) continue
                    val b = line.indexOf(',', a + 1).let { if (it < 0) line.length else it }
                    val mv = line.substring(a + 1, b).toFloatOrNull() ?: continue // HR / marker-only row
                    if (n == t.size) {
                        t = t.copyOf(n * 2)
                        v = v.copyOf(n * 2)
                    }
                    t[n] = ts
                    v[n] = mv
                    n++
                }
            }
        }
        // Files are written in time order, but sort in case of an append after a clock change.
        val order = (0 until n).sortedBy { t[it] }
        return EcgWindow(LongArray(n) { t[order[it]] }, FloatArray(n) { v[order[it]] })
    }

    /**
     * The H10's heart rate in bins of [binNs] for [fromNs] <= t < [toNs], from the
     * hourly `hr_*.csv` files. Readings without skin contact (when the strap reports
     * contact) are left out: the H10 keeps sending made-up values when it is off.
     */
    fun heartRate(root: File, zone: ZoneId, fromNs: Long, toNs: Long, binNs: Long = 60_000_000_000L): List<HrBin> {
        val bins = sortedMapOf<Long, MutableList<Int>>()
        var lastTs = Long.MIN_VALUE
        for (file in hourlyFiles(fromNs, toNs, zone) { RecordingFiles.hrFile(root, it) }) {
            file.bufferedReader().useLines { lines ->
                for (line in lines) {
                    val c = line.split(',')
                    if (c.size < 5) continue
                    val ts = c[0].toLongOrNull() ?: continue // header
                    if (ts < fromNs || ts >= toNs) continue
                    // One row per RR interval; all rows of one notification share time and HR.
                    if (ts == lastTs) continue
                    lastTs = ts
                    val hr = c[1].toIntOrNull() ?: continue
                    val contact = c[3] == "1"
                    val supported = c[4] == "1"
                    if (hr <= 0 || (supported && !contact)) continue
                    val bin = fromNs + (ts - fromNs) / binNs * binNs
                    bins.getOrPut(bin) { mutableListOf() }.add(hr)
                }
            }
        }
        return bins.map { (start, values) ->
            values.sort()
            HrBin(start, values[values.size / 2], values.first(), values.last(), values.size)
        }
    }

    private fun hourlyFiles(fromNs: Long, toNs: Long, zone: ZoneId, fileOf: (java.time.LocalDateTime) -> File): List<File> {
        if (toNs <= fromNs) return emptyList()
        val out = mutableListOf<File>()
        var hour = RecordingFiles.hourOf(fromNs, zone)
        val last = RecordingFiles.hourOf(toNs - 1, zone)
        while (!hour.isAfter(last)) {
            fileOf(hour).takeIf { it.exists() }?.let { out += it }
            hour = hour.plusHours(1)
        }
        return out
    }
}
