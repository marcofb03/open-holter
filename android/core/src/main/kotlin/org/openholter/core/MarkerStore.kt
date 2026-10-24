package org.openholter.core

import java.io.File
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/** A symptom marker: the phone time of the button press and an optional comment. */
data class Marker(val timestampNs: Long, val comment: String = "")

/**
 * Symptom markers with optional comments, one small CSV per day
 * (`markers_<day>.csv`, columns [RecordingFiles.MARKERS_HEADER]; the comment is
 * always quoted, `"` doubled).
 *
 * The press is also written to the ECG file (`marker` column) while recording,
 * and to the event log. This file adds the comment and also keeps presses made
 * while no ECG was being recorded. Markers from before this file existed are
 * read from `events_<day>.log`.
 */
class MarkerStore(private val root: File, private val zone: ZoneId) {

    /** Records a press. Keeps the comment if the marker is already there. */
    @Synchronized
    fun add(timestampNs: Long) = upsert(timestampNs, null)

    /** Sets (or replaces) the comment; adds the marker if it is not stored yet. */
    @Synchronized
    fun setComment(timestampNs: Long, comment: String) = upsert(timestampNs, clean(comment))

    /** All markers of one local day, oldest first. */
    @Synchronized
    fun list(day: LocalDate): List<Marker> {
        val stored = read(day)
        val legacy = legacyMarkers(day).filter { t -> stored.none { Math.abs(it.timestampNs - t) < MATCH_NS } }
        return (stored + legacy.map { Marker(it) }).sortedBy { it.timestampNs }
    }

    /** Number of markers per day for the days in [from]..[to] that have any. */
    @Synchronized
    fun countsByDay(from: LocalDate, to: LocalDate): Map<LocalDate, Int> =
        RecordingFiles.listDays(root).filter { it in from..to }
            .associateWith { list(it).size }
            .filterValues { it > 0 }

    private fun upsert(timestampNs: Long, comment: String?) {
        val day = RecordingFiles.dayOf(timestampNs, zone)
        val markers = read(day).toMutableList()
        val i = markers.indexOfFirst { Math.abs(it.timestampNs - timestampNs) < MATCH_NS }
        when {
            i < 0 -> markers += Marker(timestampNs, comment.orEmpty())
            comment != null -> markers[i] = markers[i].copy(comment = comment)
            else -> return
        }
        write(day, markers.sortedBy { it.timestampNs })
    }

    private fun read(day: LocalDate): List<Marker> {
        val file = RecordingFiles.markersFile(root, day)
        if (!file.exists()) return emptyList()
        return file.readLines().mapNotNull { parse(it) }
    }

    private fun write(day: LocalDate, markers: List<Marker>) {
        val file = RecordingFiles.markersFile(root, day)
        file.parentFile?.mkdirs()
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(
            buildString {
                append(RecordingFiles.MARKERS_HEADER).append('\n')
                for (m in markers) {
                    val local = LocalDateTime.ofInstant(RecordingFiles.instantOf(m.timestampNs), zone)
                    append(m.timestampNs).append(',').append(FORMAT.format(local)).append(',')
                    append('"').append(m.comment.replace("\"", "\"\"")).append('"').append('\n')
                }
            },
        )
        if (!tmp.renameTo(file)) {
            file.delete()
            tmp.renameTo(file)
        }
    }

    /** Presses logged as `marker` in the day's event log. */
    private fun legacyMarkers(day: LocalDate): List<Long> {
        val file = RecordingFiles.eventsFile(root, day)
        if (!file.exists()) return emptyList()
        return file.readLines().mapNotNull { line ->
            val cols = line.split('\t')
            if (cols.size >= 3 && cols[2] == "marker") cols[1].toLongOrNull() else null
        }
    }

    companion object {
        /** Presses closer than this are the same marker (event log and file differ by microseconds). */
        private const val MATCH_NS = 5_000_000L
        private val FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")

        /** One line, no control characters (keeps the CSV simple for any reader). */
        fun clean(comment: String): String =
            comment.map { if (it.isISOControl()) ' ' else it }.joinToString("").trim()

        internal fun parse(line: String): Marker? {
            val a = line.indexOf(',')
            if (a < 0) return null
            val ts = line.substring(0, a).toLongOrNull() ?: return null // header
            val b = line.indexOf(',', a + 1)
            if (b < 0) return Marker(ts)
            val raw = line.substring(b + 1)
            val comment = if (raw.length >= 2 && raw.startsWith('"') && raw.endsWith('"')) {
                raw.substring(1, raw.length - 1).replace("\"\"", "\"")
            } else {
                raw
            }
            return Marker(ts, comment)
        }
    }
}
