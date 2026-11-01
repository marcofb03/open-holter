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

    @Test
    fun `extreme finds negative waves too, and amplitude labels are point measurements`() {
        val mv = FloatArray(260) { 0.1f }
        mv[60] = 1.5f
        mv[63] = -0.9f
        val w = EcgWindow(LongArray(260) { t0 + it * step }, mv)
        assertEquals(t0 + 60 * step, EcgSnap.extreme(w, t0 + 61 * step, baseline = 0.1f))
        assertEquals(t0 + 63 * step, EcgSnap.extreme(w, t0 + 63 * step, baseline = 0.1f, radiusNs = 10_000_000L))
        assertEquals(true, Measurement(t0, t0).isAmplitude)
        assertEquals(false, Measurement(t0, t0 + 1).isAmplitude)
    }

    @Test
    fun `row baseline and extent above and below it`() {
        // Two 1-s rows: baseline 0.2 mV with a +1.3 mV peak, then baseline -0.1 mV with a -0.8 mV trough.
        val n = 260
        val mv = FloatArray(n) { if (it < 130) 0.2f else -0.1f }
        mv[40] = 1.5f
        mv[200] = -0.9f
        val w = EcgWindow(LongArray(n) { t0 + it * step }, mv)
        assertEquals(0.2f, EcgRows.median(w, t0, t0 + 1_000_000_000L)!!, 1e-6f)
        val (up, down) = EcgRows.extent(w, t0, t0 + 2_000_000_000L, 1_000_000_000L)
        assertEquals(1.3f, up, 1e-5f)
        assertEquals(0.8f, down, 1e-5f)
        assertNull(EcgRows.median(w, t0 + 10_000_000_000L, t0 + 11_000_000_000L))
    }

    @Test
    fun `snap takes the top or the bottom nearest to the tap, not simply the biggest`() {
        // Baseline 0, an R spike of +2 mV at sample 60 and an S dip of -0.8 mV 4 samples (31 ms) later,
        // with small noise wiggles of 0.05 mV around them.
        val n = 260
        val mv = FloatArray(n) { if (it % 2 == 0) 0.05f else 0f }
        for (k in 55..59) mv[k] = 0.3f * (k - 54) // rising edge up to 1.5 mV, below the 2 mV tip
        mv[60] = 2f
        mv[61] = 1f; mv[62] = 0f; mv[63] = -0.5f; mv[64] = -0.8f; mv[65] = -0.4f; mv[66] = 0f
        val w = EcgWindow(LongArray(n) { t0 + it * step }, mv)
        val nsPerPx = 4e6 // 4 ms per pixel
        val mvPerPx = 0.02 // 50 px per mV
        // Tap just under the dip, 1 sample to its right: the bottom of the dip, although the spike is bigger.
        assertEquals(EcgSnap.Extremum(t0 + 64 * step, false), EcgSnap.nearestExtremum(w, t0 + 65 * step, -0.9f, nsPerPx, mvPerPx))
        // Tap above the spike, 100 ms to its left (outside the old 50 ms): the tip of the spike.
        assertEquals(EcgSnap.Extremum(t0 + 60 * step, true), EcgSnap.nearestExtremum(w, t0 + 47 * step, 2.1f, nsPerPx, mvPerPx))
        // Noise wiggles are not tops or bottoms.
        assertNull(EcgSnap.kind(w, t0 + 200 * step))
        assertEquals(true, EcgSnap.kind(w, t0 + 60 * step))
        assertEquals(false, EcgSnap.kind(w, t0 + 64 * step))
    }
}
