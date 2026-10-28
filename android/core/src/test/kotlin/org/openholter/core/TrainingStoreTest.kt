package org.openholter.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset

class TrainingStoreTest {
    @get:Rule val tmp = TemporaryFolder()

    private val zone = ZoneOffset.UTC
    private val day = LocalDate.of(2026, 10, 28)
    private val t0 = LocalDateTime.of(2026, 10, 28, 17, 0).toEpochSecond(ZoneOffset.UTC) * 1_000_000_000L
    private val min = 60_000_000_000L

    @Test
    fun `start, stop and comment round-trip through the file`() {
        val s = TrainingStore(tmp.root, zone)
        s.start(t0)
        assertEquals(Training(t0), s.running(day))
        s.stop(t0, t0 + 45 * min)
        s.setComment(t0, "tennis, \"doubles\"")
        assertNull(s.running(day))
        assertEquals(listOf(Training(t0, t0 + 45 * min, "tennis, \"doubles\"")), TrainingStore(tmp.root, zone).list(day))
        val lines = RecordingFiles.trainingsFile(tmp.root, day).readLines()
        assertEquals(RecordingFiles.TRAININGS_HEADER, lines[0])
        assertEquals(
            "$t0,${t0 + 45 * min},2026-10-28 17:00:00.000,2026-10-28 17:45:00.000,\"tennis, \"\"doubles\"\"\"",
            lines[1],
        )
    }

    @Test
    fun `a running training survives a restart and is found from the next day`() {
        val late = t0 + 6 * 60 * min + 30 * min // 23:30
        TrainingStore(tmp.root, zone).start(late)
        val again = TrainingStore(tmp.root, zone)
        assertEquals(Training(late), again.running(day.plusDays(1)))
        assertEquals("", RecordingFiles.trainingsFile(tmp.root, day).readLines()[1].split(',')[1])
    }

    @Test
    fun `counts per day and end never before start`() {
        val s = TrainingStore(tmp.root, zone)
        s.start(t0)
        s.stop(t0, t0 - min)
        s.start(t0 + 120 * min)
        assertEquals(t0, s.list(day)[0].endNs)
        assertEquals(mapOf(day to 2), s.countsByDay(day.minusDays(1), day.plusDays(1)))
    }
}
