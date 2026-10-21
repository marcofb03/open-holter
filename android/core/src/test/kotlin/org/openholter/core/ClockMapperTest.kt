package org.openholter.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ClockMapperTest {
    private val period = 1_000_000_000L / 130
    private val unix0 = 1_792_000_000_000_000_000L
    private val dev0 = 846_000_000_000_000_000L // device clock, epoch 2000

    @Test
    fun `first packet sets offset from arrival`() {
        val m = ClockMapper()
        m.onPacket(unix0 + 50_000_000L, dev0)
        assertEquals(unix0 + 50_000_000L, m.map(dev0))
    }

    @Test
    fun `offset converges to the lowest latency slowly and keeps spacing`() {
        val m = ClockMapper(maxSlewNs = 2_000_000L)
        val trueOffset = unix0 - dev0
        val latencies = longArrayOf(80, 40, 60, 20, 30, 25, 20, 90, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20).map { it * 1_000_000L }
        var dev = dev0
        var last = Long.MIN_VALUE
        for (lat in latencies) {
            val samples = (0 until 73).map { dev + it * period }
            val lastSample = samples.last()
            m.onPacket(lastSample + trueOffset + lat, lastSample)
            for (s in samples) {
                val t = m.map(s)
                assertTrue("timestamps must increase", t > last)
                if (last != Long.MIN_VALUE) assertTrue("spacing stays close to 1/130 s", t - last in (period - 2_000_000L)..(period + 2_000_000L))
                last = t
            }
            dev += 73 * period
        }
        // Converged to the minimum latency seen (20 ms).
        assertEquals(trueOffset + 20_000_000L, m.offsetNs)
    }

    @Test
    fun `follows slow clock drift`() {
        val m = ClockMapper(windowNs = 10_000_000_000L)
        val drift = 20e-6 // device clock 20 ppm slow
        var dev = dev0
        repeat(20_000) { // ~3 hours of packets every 0.56 s
            val elapsed = dev - dev0
            val arrival = unix0 + (elapsed * (1 + drift)).toLong() + 15_000_000L
            m.onPacket(arrival, dev)
            m.map(dev)
            dev += 73 * period
        }
        val elapsed = dev - 73 * period - dev0
        val expected = unix0 + (elapsed * (1 + drift)).toLong() + 15_000_000L
        val mapped = (dev - 73 * period) + m.offsetNs!!
        assertTrue("error ${(mapped - expected) / 1e6} ms", Math.abs(mapped - expected) < 5_000_000L)
    }

    @Test
    fun `device clock reset resyncs immediately and output stays increasing`() {
        val m = ClockMapper()
        m.onPacket(unix0, dev0)
        val a = m.map(dev0)
        // Reconnect: device clock restarted near zero.
        m.onPacket(unix0 + 10_000_000_000L, 1_000_000L)
        val b = m.map(1_000_000L)
        assertEquals(unix0 + 10_000_000_000L, b)
        assertTrue(b > a)
    }
}
