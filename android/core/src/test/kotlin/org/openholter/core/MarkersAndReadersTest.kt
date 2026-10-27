package org.openholter.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset

class MarkersAndReadersTest {
    @get:Rule val tmp = TemporaryFolder()

    private val zone = ZoneOffset.UTC
    private val day = LocalDate.of(2026, 10, 24)
    private val hour = LocalDateTime.of(2026, 10, 24, 9, 0)
    private val t0 = hour.toEpochSecond(ZoneOffset.UTC) * 1_000_000_000L

    @Test
    fun `markers keep comments, quotes and order`() {
        val s = MarkerStore(tmp.root, zone)
        s.add(t0 + 2_000_000_000L)
        s.add(t0)
        s.setComment(t0, "palpitations, after \"stairs\"\nshort")
        s.add(t0) // a second add must not erase the comment
        assertEquals(
            listOf(Marker(t0, "palpitations, after \"stairs\" short"), Marker(t0 + 2_000_000_000L, "")),
            MarkerStore(tmp.root, zone).list(day),
        )
        val lines = RecordingFiles.markersFile(tmp.root, day).readLines()
        assertEquals(RecordingFiles.MARKERS_HEADER, lines[0])
        assertEquals("$t0,2026-10-24 09:00:00.000,\"palpitations, after \"\"stairs\"\" short\"", lines[1])
    }

    @Test
    fun `comment for a marker that is not stored yet adds it`() {
        val s = MarkerStore(tmp.root, zone)
        s.setComment(t0, "dizzy")
        s.add(t0)
        assertEquals(listOf(Marker(t0, "dizzy")), s.list(day))
    }

    @Test
    fun `markers from the event log are listed and can get a comment`() {
        val log = EventLog(tmp.root, zone)
        log.log(t0, "marker", "symptom marker")
        log.log(t0 + 1_000_000_000L, "connected", "A1B2C3D4")
        val s = MarkerStore(tmp.root, zone)
        assertEquals(listOf(Marker(t0)), s.list(day))
        s.setComment(t0 + 1_000L, "old one") // same press, timestamps differ by microseconds
        assertEquals(listOf(Marker(t0 + 1_000L, "old one")), s.list(day))
        assertEquals(mapOf(day to 1), s.countsByDay(day.minusDays(3), day))
    }

    @Test
    fun `ecg window skips header, other rows and out-of-window samples`() {
        val w = EcgCsvWriter(tmp.root, zone)
        for (i in 0 until 130 * 3) w.writeEcg(t0 + i * 7_692_308L, 1000 + i)
        w.addMarker(t0 + 1_000_000_000L)
        w.close()
        val win = SignalReaders.ecg(tmp.root, zone, t0 + 1_000_000_000L, t0 + 2_000_000_000L)
        assertTrue(win.size in 129..131)
        assertTrue(win.t.all { it >= t0 + 1_000_000_000L && it < t0 + 2_000_000_000L })
        assertEquals((1000 + 130) / 1000f, win.mv[0], 0.002f)
    }

    @Test
    fun `ecg window spans two hourly files`() {
        val w = EcgCsvWriter(tmp.root, zone)
        val end = t0 + 3_600_000_000_000L
        w.writeEcg(end - 500_000_000L, 100)
        w.writeEcg(end + 500_000_000L, 200)
        w.close()
        val win = SignalReaders.ecg(tmp.root, zone, end - 1_000_000_000L, end + 1_000_000_000L)
        assertEquals(listOf(end - 500_000_000L, end + 500_000_000L), win.t.toList())
    }

    @Test
    fun `heart rate bins drop readings without skin contact and duplicate RR rows`() {
        val w = HrCsvWriter(tmp.root, zone)
        w.write(t0, 60, listOf(1000, 990), contact = true, contactSupported = true) // two rows, one reading
        w.write(t0 + 1_000_000_000L, 64, emptyList(), contact = true, contactSupported = true)
        w.write(t0 + 2_000_000_000L, 120, emptyList(), contact = false, contactSupported = true) // strap off
        w.write(t0 + 3_000_000_000L, 0, emptyList(), contact = true, contactSupported = true)
        w.write(t0 + 61_000_000_000L, 70, emptyList(), contact = true, contactSupported = true)
        w.close()
        val bins = SignalReaders.heartRate(tmp.root, zone, t0, t0 + 120_000_000_000L)
        assertEquals(
            listOf(HrBin(t0, 64, 60, 64, 2), HrBin(t0 + 60_000_000_000L, 70, 70, 70, 1)),
            bins,
        )
    }
}
