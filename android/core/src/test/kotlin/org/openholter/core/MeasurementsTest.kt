package org.openholter.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset

class MeasurementsTest {
    @get:Rule val tmp = TemporaryFolder()

    private val zone = ZoneOffset.UTC
    private val t0 = LocalDateTime.of(2026, 11, 1, 10, 0).toEpochSecond(ZoneOffset.UTC) * 1_000_000_000L
    private val step = 7_692_308L // 130 Hz

    /** 2 s of flat ECG with a 2 mV peak at sample 60 and a 1 mV bump at sample 64. */
    private fun window(): EcgWindow {
        val n = 260
        val mv = FloatArray(n) { 0f }
        mv[60] = 2f
        mv[64] = 1f
        return EcgWindow(LongArray(n) { t0 + it * step }, mv)
    }

    @Test
    fun `peak snaps to the highest sample within 50 ms`() {
        val w = window()
        assertEquals(t0 + 60 * step, EcgSnap.peak(w, t0 + 64 * step)) // 31 ms away: the bigger peak wins
        assertEquals(t0 + 64 * step, EcgSnap.peak(w, t0 + 64 * step, radiusNs = 20_000_000L))
        // On a flat stretch any sample within 50 ms is acceptable.
        val flat = EcgSnap.peak(w, t0 + 200 * step)!!
        assertEquals(true, Math.abs(flat - (t0 + 200 * step)) <= 50_000_000L)
    }

    @Test
    fun `free placement takes the nearest sample`() {
        val w = window()
        assertEquals(t0 + 10 * step, EcgSnap.nearest(w, t0 + 10 * step + step / 3))
        assertEquals(t0 + 11 * step, EcgSnap.nearest(w, t0 + 10 * step + 2 * step / 3))
        assertEquals(t0, EcgSnap.nearest(w, t0 - 1_000_000_000L))
        assertEquals(2f, EcgSnap.valueAt(w, t0 + 60 * step))
        assertNull(EcgSnap.peak(EcgWindow(LongArray(0), FloatArray(0)), t0))
    }

    @Test
    fun `measurements per marker round-trip and can be removed`() {
        val s = MeasurementStore(tmp.root, zone)
        val other = t0 + 3_600_000_000_000L
        val a = Measurement(t0, t0 + 812_000_000L)
        val b = Measurement(t0 + 900_000_000L, t0 + 1_000_000_000L)
        s.add(t0, a)
        s.add(t0, b)
        s.add(other, a)
        assertEquals(listOf(a, b), MeasurementStore(tmp.root, zone).list(t0))
        s.remove(t0, a)
        assertEquals(listOf(b), s.list(t0))
        assertEquals(listOf(a), s.list(other))
        assertEquals(812.0, a.intervalMs, 1e-9)
        assertEquals(73.9, a.bpm, 0.05)
        val lines = RecordingFiles.measurementsFile(tmp.root, LocalDate.of(2026, 11, 1)).readLines()
        assertEquals(RecordingFiles.MEASUREMENTS_HEADER, lines[0])
        assertEquals("$t0,${t0 + 900_000_000L},${t0 + 1_000_000_000L},100.0", lines[1])
    }
}
