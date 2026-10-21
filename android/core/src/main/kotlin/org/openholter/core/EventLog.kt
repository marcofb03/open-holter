package org.openholter.core

import java.io.File
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Human-readable, tab-separated log of what happened during recording
 * (connects, drop-outs, ECG gaps, pause/resume, markers). One file per day.
 * Each line is written and flushed immediately; volume is tiny.
 */
class EventLog(private val root: File, private val zone: ZoneId) {

    fun log(timestampNs: Long, event: String, detail: String = "") {
        val file = RecordingFiles.eventsFile(root, RecordingFiles.dayOf(timestampNs, zone))
        file.parentFile?.mkdirs()
        val isNew = !file.exists() || file.length() == 0L
        val local = LocalDateTime.ofInstant(RecordingFiles.instantOf(timestampNs), zone)
        val clean = detail.replace('\t', ' ').replace('\n', ' ')
        file.appendText(
            buildString {
                if (isNew) append(RecordingFiles.EVENTS_HEADER).append('\n')
                append(FORMAT.format(local)).append('\t')
                append(timestampNs).append('\t')
                append(event).append('\t')
                append(clean).append('\n')
            },
        )
    }

    private companion object {
        val FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")
    }
}
