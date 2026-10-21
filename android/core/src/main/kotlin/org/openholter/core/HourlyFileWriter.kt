package org.openholter.core

import java.io.BufferedWriter
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.time.LocalDateTime
import java.time.ZoneId

/**
 * Appends text lines to one file per local hour. The hour is taken from the
 * timestamp of each line, so files always contain the data of their hour.
 *
 * Existing files are appended to (e.g. after an app restart within the same
 * hour); the header is written only when a file is new or empty.
 *
 * Buffered data is handed to the OS every [flushIntervalNs] (survives an app
 * crash) and fsync'ed to flash every [syncIntervalNs] (survives a power loss).
 */
class HourlyFileWriter(
    private val zone: ZoneId,
    private val header: String,
    private val fileForHour: (LocalDateTime) -> File,
    private val flushIntervalNs: Long = 10_000_000_000L,
    private val syncIntervalNs: Long = 60_000_000_000L,
) : Closeable {

    private var hour: LocalDateTime? = null
    private var stream: FileOutputStream? = null
    private var writer: BufferedWriter? = null
    private var lastFlushNs = Long.MIN_VALUE
    private var lastSyncNs = Long.MIN_VALUE
    private var dirty = false

    /** File currently being written, or null before the first line / after [close]. */
    var currentFile: File? = null
        private set

    fun writeLine(timestampNs: Long, line: CharSequence) {
        val lineHour = RecordingFiles.hourOf(timestampNs, zone)
        if (lineHour != hour) open(lineHour, timestampNs)
        writer!!.append(line).append('\n')
        dirty = true
    }

    /** Call regularly (e.g. every second) with the current wall-clock time. */
    fun tick(nowNs: Long) {
        if (writer == null) return
        if (due(lastFlushNs, nowNs, flushIntervalNs)) {
            val sync = due(lastSyncNs, nowNs, syncIntervalNs)
            flush(sync)
            lastFlushNs = nowNs
            if (sync) lastSyncNs = nowNs
        }
    }

    fun flush(sync: Boolean = false) {
        val w = writer ?: return
        if (dirty) {
            w.flush()
            dirty = false
        }
        if (sync) runCatching { stream?.fd?.sync() }
    }

    override fun close() {
        flush(sync = true)
        writer?.close()
        writer = null
        stream = null
        hour = null
        currentFile = null
    }

    private fun open(newHour: LocalDateTime, timestampNs: Long) {
        close()
        val file = fileForHour(newHour)
        file.parentFile?.mkdirs()
        val isNew = !file.exists() || file.length() == 0L
        val out = FileOutputStream(file, true)
        val w = BufferedWriter(OutputStreamWriter(out, Charsets.UTF_8), 64 * 1024)
        if (isNew) w.append(header).append('\n')
        stream = out
        writer = w
        hour = newHour
        currentFile = file
        dirty = true
        lastFlushNs = timestampNs
        lastSyncNs = timestampNs
    }

    private fun due(last: Long, now: Long, interval: Long): Boolean =
        last == Long.MIN_VALUE || now - last >= interval || now < last
}
