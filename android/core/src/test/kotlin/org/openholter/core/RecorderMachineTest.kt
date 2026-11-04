package org.openholter.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.openholter.core.RecorderMachine.Action
import org.openholter.core.RecorderMachine.Event
import org.openholter.core.RecorderMachine.State

class RecorderMachineTest {
    private val id = "A1B2C3D4"
    private var now = 1_000_000L
    private val m = RecorderMachine()

    private fun send(e: Event): List<Action> = m.handle(e, now)
    private fun List<Action>.ble() = filterNot { it is Action.Log }

    private fun streaming() {
        send(Event.Start(id))
        send(Event.DeviceConnected(id))
        send(Event.StreamingReady(id))
        assertTrue(m.state is State.Streaming)
    }

    /** Polar known issue: a disconnect must never happen while ECG/ACC may still be streaming. */
    private fun assertStopsBeforeDisconnect(actions: List<Action>) {
        val d = actions.indexOfFirst { it is Action.Disconnect }
        if (d < 0) return
        val s = actions.indexOfFirst { it is Action.StopStreams }
        assertTrue("StopStreams must come before Disconnect: $actions", s in 0 until d)
    }

    @Test
    fun `start connects and streams once the feature is ready`() {
        assertEquals(listOf(Action.Connect(id)), send(Event.Start(id)).ble())
        assertTrue(m.state is State.Connecting)
        assertEquals(emptyList<Action>(), send(Event.DeviceConnected(id)).ble())
        assertTrue(m.state is State.Connected)
        assertEquals(listOf(Action.StartStreams(id)), send(Event.StreamingReady(id)).ble())
        assertTrue(m.state is State.Streaming)
    }

    @Test
    fun `pause stops streams before disconnecting and resume reconnects`() {
        streaming()
        val pause = send(Event.Pause).ble()
        assertEquals(listOf(Action.StopStreams(id), Action.Disconnect(id)), pause)
        assertTrue(m.state is State.Paused)
        // The disconnect callback while paused changes nothing.
        assertEquals(emptyList<Action>(), send(Event.DeviceDisconnected(id)).ble())
        // No reconnect attempts while paused, however long it lasts.
        now += 3_600_000
        assertEquals(emptyList<Action>(), send(Event.Tick).ble())
        assertEquals(listOf(Action.Connect(id)), send(Event.Resume).ble())
        assertTrue(m.state is State.Connecting)
    }

    @Test
    fun `stop from streaming stops streams first`() {
        streaming()
        val actions = send(Event.Stop)
        assertStopsBeforeDisconnect(actions)
        assertEquals(listOf(Action.StopStreams(id), Action.Disconnect(id)), actions.ble())
        assertEquals(State.Idle, m.state)
    }

    @Test
    fun `drop-out waits for SDK reconnect then kicks with back-off`() {
        streaming()
        send(Event.DeviceDisconnected(id))
        assertTrue(m.state is State.Connecting)
        now += 29_000
        assertEquals(emptyList<Action>(), send(Event.Tick).ble())
        now += 1_000
        assertEquals(listOf(Action.Connect(id)), send(Event.Tick).ble())
        now += 59_000
        assertEquals(emptyList<Action>(), send(Event.Tick).ble())
        now += 1_000
        assertEquals(listOf(Action.Connect(id)), send(Event.Tick).ble())
        // Strap back on: SDK connects, streams restart.
        send(Event.DeviceConnected(id))
        assertEquals(listOf(Action.StartStreams(id)), send(Event.StreamingReady(id)).ble())
    }

    @Test
    fun `back-off is capped`() {
        val p = ReconnectPolicy(listOf(1, 2, 3))
        assertEquals(1, p.delayMs(0))
        assertEquals(3, p.delayMs(2))
        assertEquals(3, p.delayMs(50))
    }

    @Test
    fun `no data restarts streams then reconnects`() {
        streaming()
        now += 20_000
        assertEquals(listOf(Action.StopStreams(id), Action.StartStreams(id)), send(Event.Tick).ble())
        now += 20_000
        assertEquals(listOf(Action.StopStreams(id), Action.StartStreams(id)), send(Event.Tick).ble())
        now += 20_000
        val actions = send(Event.Tick)
        assertStopsBeforeDisconnect(actions)
        assertEquals(listOf(Action.StopStreams(id), Action.Disconnect(id), Action.Connect(id)), actions.ble())
        assertTrue(m.state is State.Connecting)
    }

    @Test
    fun `data resets the no-data watchdog`() {
        streaming()
        repeat(10) {
            now += 10_000
            send(Event.EcgData)
            assertEquals(emptyList<Action>(), send(Event.Tick).ble())
        }
    }

    @Test
    fun `long loss of skin contact stops streams and contact restarts them`() {
        streaming()
        send(Event.Contact(false))
        now += 60_000
        send(Event.EcgData)
        send(Event.Contact(false))
        assertEquals(emptyList<Action>(), send(Event.Tick).ble())
        now += 120_000
        send(Event.EcgData)
        assertEquals(listOf(Action.StopStreams(id)), send(Event.Tick).ble())
        assertTrue(m.state is State.NoContact)
        assertEquals(listOf(Action.StartStreams(id)), send(Event.Contact(true)).ble())
        assertTrue(m.state is State.Streaming)
    }

    @Test
    fun `unsupported contact sample does not restart streams`() {
        // Seen on a real H10 (Nov 2026): right after the streams were stopped it sent
        // contact=false, supported=false. That used to count as "contact back" and
        // restarted the streams every ~3 min all night with the strap off.
        streaming()
        send(Event.Contact(false))
        now += 180_000
        send(Event.Tick)
        assertTrue(m.state is State.NoContact)
        assertEquals(emptyList<Action>(), send(Event.Contact(false, supported = false)).ble())
        assertEquals(emptyList<Action>(), send(Event.Contact(true, supported = false)).ble())
        assertEquals(emptyList<Action>(), send(Event.Contact(false)).ble())
        assertTrue(m.state is State.NoContact)
        // Real contact restarts.
        assertEquals(listOf(Action.StartStreams(id)), send(Event.Contact(true)).ble())
        assertTrue(m.state is State.Streaming)
    }

    @Test
    fun `unsupported contact sample does not reset the no-contact timer`() {
        streaming()
        send(Event.Contact(false))
        now += 90_000
        send(Event.EcgData)
        send(Event.Contact(false, supported = false))
        send(Event.Contact(false))
        now += 90_000
        send(Event.EcgData)
        assertEquals(listOf(Action.StopStreams(id)), send(Event.Tick).ble())
        assertTrue(m.state is State.NoContact)
    }

    @Test
    fun `strap off all night stops streams once and does not loop`() {
        // Replays the pattern of the Nov 3-4 night: contact=false about once a second
        // with an occasional supported=false sample, for 8 hours.
        streaming()
        var starts = 0
        var stops = 0
        repeat(8 * 3600) { k ->
            now += 1_000
            val e = if (k % 80 == 0) Event.Contact(false, supported = false) else Event.Contact(false)
            for (a in send(Event.EcgData) + send(e) + send(Event.Tick)) {
                if (a is Action.StartStreams) starts++
                if (a is Action.StopStreams) stops++
            }
        }
        assertEquals(1, stops)
        assertEquals(0, starts)
        assertTrue(m.state is State.NoContact)
    }

    @Test
    fun `strap without contact support keeps streaming`() {
        streaming()
        repeat(600) {
            now += 1_000
            send(Event.EcgData)
            send(Event.Contact(false, supported = false))
            assertEquals(emptyList<Action>(), send(Event.Tick).ble())
        }
        assertTrue(m.state is State.Streaming)
    }

    @Test
    fun `brief contact loss is tolerated`() {
        streaming()
        send(Event.Contact(false))
        now += 100_000
        send(Event.EcgData)
        send(Event.Contact(true))
        now += 100_000
        send(Event.EcgData)
        assertEquals(emptyList<Action>(), send(Event.Tick).ble())
        assertTrue(m.state is State.Streaming)
    }

    @Test
    fun `unexpected connection while paused or idle is closed`() {
        assertEquals(listOf(Action.Disconnect(id)), send(Event.DeviceConnected(id)).ble())
        streaming()
        send(Event.Pause)
        assertEquals(listOf(Action.Disconnect(id)), send(Event.DeviceConnected(id)).ble())
        assertTrue(m.state is State.Paused)
    }

    @Test
    fun `restore after reboot keeps pause`() {
        assertEquals(emptyList<Action>(), send(Event.Start(id, paused = true)).ble())
        assertTrue(m.state is State.Paused)
        // Starting again (not paused) resumes.
        assertEquals(listOf(Action.Connect(id)), send(Event.Start(id)).ble())
    }

    @Test
    fun `switching device tears down the old one cleanly`() {
        streaming()
        val actions = send(Event.Start("ZZZZ0000"))
        assertStopsBeforeDisconnect(actions)
        assertEquals(
            listOf(Action.StopStreams(id), Action.Disconnect(id), Action.Connect("ZZZZ0000")),
            actions.ble(),
        )
    }

    @Test
    fun `bluetooth off and on`() {
        streaming()
        assertEquals(listOf(Action.StopStreams(id)), send(Event.BluetoothPower(false)).ble())
        assertTrue(m.state is State.Connecting)
        assertEquals(listOf(Action.Connect(id)), send(Event.BluetoothPower(true)).ble())
    }

    @Test
    fun `never disconnects while streams are running, whatever happens`() {
        val events = listOf(
            Event.Pause, Event.Resume, Event.Stop, Event.Start(id), Event.Start("OTHER123"),
            Event.DeviceConnected(id), Event.DeviceDisconnected(id), Event.StreamingReady(id), Event.EcgData,
            Event.Contact(false), Event.Contact(true), Event.Contact(false, supported = false),
            Event.BluetoothPower(false), Event.BluetoothPower(true),
            Event.Tick,
        )
        val rnd = java.util.Random(42)
        val machine = RecorderMachine()
        val running = mutableSetOf<String>()
        var t = 0L
        repeat(50_000) {
            t += rnd.nextInt(60_000)
            for (a in machine.handle(events[rnd.nextInt(events.size)], t)) {
                when (a) {
                    is Action.StartStreams -> running += a.deviceId
                    is Action.StopStreams -> running -= a.deviceId
                    is Action.Disconnect -> assertTrue("disconnect while streaming", a.deviceId !in running)
                    else -> Unit
                }
            }
            // Streams only run in the Streaming state.
            val s = machine.state
            if (s is State.Streaming) assertEquals(setOf(s.deviceId), running) else assertEquals(emptySet<String>(), running)
        }
    }
}
