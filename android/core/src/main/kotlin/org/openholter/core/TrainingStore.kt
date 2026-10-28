package org.openholter.core

import java.io.File
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/** A training (or any activity the owner wants to track): start, end (null while running), comment. */
data class Training(val startNs: Long, val endNs: Long? = null, val comment: String = "") {
    val running: Boolean get() = endNs == null
}

/**
 * Training sessions, one small CSV per day of the start
 * (`trainings_<day>.csv`, columns [RecordingFiles.TRAININGS_HEADER]; `end_ns`
 * is empty while the training runs, the comment is always quoted).
 * A running training is in the file from its start, so it survives an app
 * restart or a reboot.
 */
class TrainingStore(private val root: File, private val zone: ZoneId) {

    @Synchronized
    fun start(startNs: Long): Training {
        val t = Training(startNs)
        val day = RecordingFiles.dayOf(startNs, zone)
        val list = read(day).filter { it.startNs != startNs } + t
        write(day, list)
        return t
    }

    /** Ends the training that started at [startNs]. */
    @Synchronized
    fun stop(startNs: Long, endNs: Long) = update(startNs) { it.copy(endNs = maxOf(endNs, startNs)) }

    @Synchronized
    fun setComment(startNs: Long, comment: String) = update(startNs) { it.copy(comment = MarkerStore.clean(comment)) }

    /** Trainings that started on [day], oldest first. */
    @Synchronized
    fun list(day: LocalDate): List<Training> = read(day)

    /** Number of trainings per start day for the days in [from]..[to] that have any. */
    @Synchronized
    fun countsByDay(from: LocalDate, to: LocalDate): Map<LocalDate, Int> =
        RecordingFiles.listDays(root).filter { it in from..to }
            .associateWith { read(it).size }
            .filterValues { it > 0 }

    /** The newest training without an end, started within the last [days] days. */
    @Synchronized
    fun running(today: LocalDate, days: Int = 3): Training? =
        (0 until days).asSequence()
            .flatMap { read(today.minusDays(it.toLong())).asReversed().asSequence() }
            .firstOrNull { it.running }

    private fun update(startNs: Long, change: (Training) -> Training) {
        val day = RecordingFiles.dayOf(startNs, zone)
        val list = read(day)
        val i = list.indexOfFirst { it.startNs == startNs }
        if (i < 0) return
        write(day, list.toMutableList().also { it[i] = change(it[i]) })
    }

    private fun read(day: LocalDate): List<Training> {
        val file = RecordingFiles.trainingsFile(root, day)
        if (!file.exists()) return emptyList()
        return file.readLines().mapNotNull { parse(it) }.sortedBy { it.startNs }
    }

    private fun write(day: LocalDate, list: List<Training>) {
        val file = RecordingFiles.trainingsFile(root, day)
        file.parentFile?.mkdirs()
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(
            buildString {
                append(RecordingFiles.TRAININGS_HEADER).append('\n')
                for (t in list.sortedBy { it.startNs }) {
                    append(t.startNs).append(',').append(t.endNs?.toString().orEmpty()).append(',')
                    append(local(t.startNs)).append(',').append(t.endNs?.let { local(it) }.orEmpty()).append(',')
                    append('"').append(t.comment.replace("\"", "\"\"")).append('"').append('\n')
                }
            },
        )
        if (!tmp.renameTo(file)) {
            file.delete()
            tmp.renameTo(file)
        }
    }

    private fun local(ns: Long): String = FORMAT.format(LocalDateTime.ofInstant(RecordingFiles.instantOf(ns), zone))

    companion object {
        private val FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")

        internal fun parse(line: String): Training? {
            val cuts = IntArray(4)
            var from = 0
            for (k in 0 until 4) {
                cuts[k] = line.indexOf(',', from)
                if (cuts[k] < 0) return null
                from = cuts[k] + 1
            }
            val start = line.substring(0, cuts[0]).toLongOrNull() ?: return null // header
            val end = line.substring(cuts[0] + 1, cuts[1]).toLongOrNull()
            val raw = line.substring(cuts[3] + 1)
            val comment = if (raw.length >= 2 && raw.startsWith('"') && raw.endsWith('"')) {
                raw.substring(1, raw.length - 1).replace("\"\"", "\"")
            } else {
                raw
            }
            return Training(start, end, comment)
        }
    }
}
