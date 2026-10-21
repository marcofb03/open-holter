package org.openholter.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.ZoneOffset

class EcgCsvWriterTest {
    @get:Rule val tmp = TemporaryFolder()

    private val zone: ZoneId = ZoneOffset.UTC
    private val period = 1_000_000_000L / 130

    /** 2026-10-21 14:00:00 UTC in ns. */
    private val t0 = LocalDateTime.of(2026, 10, 21, 14, 0).toEpochSecond(ZoneOffset.UTC) * 1_000_000_000L

    private fun rows(file: File) = file.readLines()

    @Test
    fun `header and ECGLogger columns`() {
        val w = EcgCsvWriter(tmp.root, zone)
        w.writeEcg(t0, 123)
        w.close()
        val lines = rows(RecordingFiles.ecgFile(tmp.root, LocalDateTime.of(2026, 10, 21, 14, 0)))
        assertEquals("timestamp_ns,ecg_mV,pulse_bpm,rr_ms,marker", lines[0])
        assertEquals("$t0,0.123,,,", lines[1])
        assertEquals(2, lines.size)
        assertEquals(5, lines[1].split(",").size)
    }

    @Test
    fun `microvolts to millivolts is exact`() {
        assertEquals("0.000", EcgCsvWriter.formatMillivolts(0))
        assertEquals("0.001", EcgCsvWriter.formatMillivolts(1))
        assertEquals("-0.005", EcgCsvWriter.formatMillivolts(-5))
        assertEquals("0.050", EcgCsvWriter.formatMillivolts(50))
        assertEquals("1.234", EcgCsvWriter.formatMillivolts(1234))
        assertEquals("-1.234", EcgCsvWriter.formatMillivolts(-1234))
        assertEquals("-12.000", EcgCsvWriter.formatMillivolts(-12000))
        assertEquals("-2147483.648", EcgCsvWriter.formatMillivolts(Int.MIN_VALUE))
    }

    @Test
    fun `pulse rr and marker are sparse and attached to the first row at or after the event`() {
        val w = EcgCsvWriter(tmp.root, zone)
        w.addHeartRate(t0 + 2 * period, 61)
        w.addRr(t0 + 2 * period, 980)
        w.addRr(t0 + 2 * period, 990)
        w.addMarker(t0 + 3 * period - 1)
        for (i in 0 until 6) w.writeEcg(t0 + i * period, i)
        w.close()
        val lines = rows(w.fileFor(t0)).drop(1)
        assertEquals("${t0},0.000,,,", lines[0])
        assertEquals("${t0 + period},0.001,,,", lines[1])
        assertEquals("${t0 + 2 * period},0.002,61,980,", lines[2])
        assertEquals("${t0 + 3 * period},0.003,,990,1", lines[3])
        assertEquals("${t0 + 4 * period},0.004,,,", lines[4])
    }

    @Test
    fun `events without ECG become standalone rows once stale`() {
        val w = EcgCsvWriter(tmp.root, zone, staleNs = 3_000_000_000L)
        w.writeEcg(t0, 10)
        w.addMarker(t0 + 1_000_000_000L)
        w.addHeartRate(t0 + 1_500_000_000L, 70)
        w.addRr(t0 + 1_500_000_000L, 857)
        w.tick(t0 + 2_000_000_000L) // not stale yet
        w.flush()
        assertEquals(2, rows(w.fileFor(t0)).size)
        w.tick(t0 + 5_000_000_000L)
        w.flush()
        val lines = rows(w.fileFor(t0)).drop(1)
        assertEquals("${t0 + 1_000_000_000L},,,,1", lines[1])
        assertEquals("${t0 + 1_500_000_000L},,70,857,", lines[2])
        w.close()
    }

    @Test
    fun `close writes pending events`() {
        val w = EcgCsvWriter(tmp.root, zone)
        w.writeEcg(t0, 1)
        w.addMarker(t0 + 10)
        w.close()
        assertEquals("${t0 + 10},,,,1", rows(w.fileForHour(14)).last())
    }

    @Test
    fun `several presses before one sample give one marked row`() {
        val w = EcgCsvWriter(tmp.root, zone)
        w.addMarker(t0 - 2)
        w.addMarker(t0 - 1)
        w.writeEcg(t0, 0)
        w.writeEcg(t0 + period, 0)
        w.close()
        val lines = rows(w.fileForHour(14)).drop(1)
        assertEquals(listOf("$t0,0.000,,,1", "${t0 + period},0.000,,,"), lines)
    }

    @Test
    fun `rotates hourly by sample time and appends without second header`() {
        val hour = 3_600_000_000_000L
        val w = EcgCsvWriter(tmp.root, zone)
        w.writeEcg(t0 + hour - period, 1)
        w.writeEcg(t0 + hour, 2)
        w.close()
        val h14 = RecordingFiles.ecgFile(tmp.root, LocalDateTime.of(2026, 10, 21, 14, 0))
        val h15 = RecordingFiles.ecgFile(tmp.root, LocalDateTime.of(2026, 10, 21, 15, 0))
        assertEquals("ecg_2026-10-21_14.csv", h14.name)
        assertEquals("2026-10-21", h14.parentFile.name)
        assertEquals(2, rows(h14).size)
        assertEquals(listOf(RecordingFiles.ECG_HEADER, "${t0 + hour},0.002,,,"), rows(h15))

        // App restart within the same hour: append, keep a single header.
        val again = EcgCsvWriter(tmp.root, zone)
        again.writeEcg(t0 + hour + period, 3)
        again.close()
        val lines = rows(h15)
        assertEquals(3, lines.size)
        assertEquals(1, lines.count { it == RecordingFiles.ECG_HEADER })
    }

    @Test
    fun `day folder follows the local time zone`() {
        val zurich = ZoneId.of("Europe/Zurich")
        // 2026-10-21 23:30 UTC = 2026-10-22 01:30 in Zurich (CEST, UTC+2).
        val t = LocalDateTime.of(2026, 10, 21, 23, 30).toEpochSecond(ZoneOffset.UTC) * 1_000_000_000L
        val w = EcgCsvWriter(tmp.root, zurich)
        w.writeEcg(t, 5)
        w.close()
        assertTrue(File(tmp.root, "2026-10-22/ecg_2026-10-22_01.csv").exists())
    }

    @Test
    fun `flushes buffered rows to disk on tick`() {
        val w = EcgCsvWriter(tmp.root, zone, flushIntervalNs = 10_000_000_000L)
        for (i in 0 until 130) w.writeEcg(t0 + i * period, i)
        val file = w.fileFor(t0)
        // Rows sit in the buffer until the flush interval has passed.
        w.tick(t0 + 5_000_000_000L)
        assertFalse(file.readLines().size > 1)
        w.tick(t0 + 10_000_000_000L)
        assertEquals(131, file.readLines().size)
        w.close()
    }

    @Test
    fun `reports gaps between ECG samples`() {
        val gaps = mutableListOf<Pair<Long, Long>>()
        val w = EcgCsvWriter(tmp.root, zone, onGap = { a, b -> gaps += a to b })
        w.writeEcg(t0, 0)
        w.writeEcg(t0 + period, 0)
        w.writeEcg(t0 + 5_000_000_000L, 0)
        w.close()
        assertEquals(listOf((t0 + period) to (t0 + 5_000_000_000L)), gaps)
    }

    @Test
    fun `output is readable as ECGLogger CSV`() {
        // Mirrors analysis/ecgpipe/io.py: skip lines whose first field is not a number,
        // ECG from column 1, RR from column 3, marker if column 4 is non-empty.
        val w = EcgCsvWriter(tmp.root, zone)
        w.addRr(t0, 900)
        w.addMarker(t0 + 5 * period)
        for (i in 0 until 10) w.writeEcg(t0 + i * period, -100 * i)
        w.close()
        var ecg = 0
        var rr = 0
        var markers = 0
        for (line in rows(w.fileFor(t0))) {
            val f = line.split(",")
            if (f[0].toDoubleOrNull() == null) continue
            if (f.getOrNull(1)?.toDoubleOrNull() != null) ecg++
            if (f.getOrNull(3)?.toDoubleOrNull() != null) rr++
            if (f.getOrNull(4)?.isNotBlank() == true) markers++
        }
        assertEquals(10, ecg)
        assertEquals(1, rr)
        assertEquals(1, markers)
    }

    private fun EcgCsvWriter.fileFor(t: Long) =
        RecordingFiles.ecgFile(tmp.root, RecordingFiles.hourOf(t, zone))

    private fun EcgCsvWriter.fileForHour(h: Int) =
        RecordingFiles.ecgFile(tmp.root, LocalDateTime.of(2026, 10, 21, h, 0))
}
