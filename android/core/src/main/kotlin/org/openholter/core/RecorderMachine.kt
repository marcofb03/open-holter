package org.openholter.core

/**
 * Connection / streaming logic of the recorder as a pure state machine, so it
 * can be unit-tested without Bluetooth. The Android service feeds it events
 * and executes the returned [Action]s in order.
 *
 * Rules:
 * - ECG/ACC streams are always stopped by the app ([Action.StopStreams])
 *   before any [Action.Disconnect]. Otherwise the H10 keeps streaming and
 *   drains its battery (Polar SDK known issue, H10 #2).
 * - After a drop-out the Polar SDK reconnects by itself; the machine also
 *   re-issues [Action.Connect] with back-off ([ReconnectPolicy]) in case that
 *   stalls, e.g. when the strap was off for hours.
 * - No ECG data for [Config.noDataTimeoutMs] → restart streams; after
 *   [Config.maxStreamRestarts] restarts → full reconnect.
 * - No skin contact for [Config.noContactStopMs] (strap taken off without
 *   pausing) → stop streams so the H10 can go to sleep; restart them as soon
 *   as contact is back.
 */
class RecorderMachine(
    private val config: Config = Config(),
    private val reconnect: ReconnectPolicy = ReconnectPolicy(),
) {
    data class Config(
        val noDataTimeoutMs: Long = 20_000,
        val maxStreamRestarts: Int = 2,
        val noContactStopMs: Long = 3 * 60_000,
    )

    sealed interface State {
        /** Not recording. */
        data object Idle : State

        sealed interface Active : State {
            val deviceId: String
        }

        /** Waiting for the H10 (first connect or after a drop-out). */
        data class Connecting(override val deviceId: String, val sinceMs: Long, val attempt: Int = 0) : Active

        /** Link up, waiting for the SDK's streaming feature to be ready. */
        data class Connected(override val deviceId: String, val sinceMs: Long) : Active

        /** Streams running. */
        data class Streaming(
            override val deviceId: String,
            val sinceMs: Long,
            val lastDataMs: Long,
            val noContactSinceMs: Long? = null,
            val restarts: Int = 0,
        ) : Active

        /** Link up, streams stopped because the strap has no skin contact. */
        data class NoContact(override val deviceId: String, val sinceMs: Long) : Active

        /** Paused by the user (e.g. shower): streams stopped and disconnected. */
        data class Paused(override val deviceId: String, val sinceMs: Long) : Active
    }

    sealed interface Event {
        data class Start(val deviceId: String, val paused: Boolean = false) : Event
        data object Stop : Event
        data object Pause : Event
        data object Resume : Event
        data class DeviceConnected(val deviceId: String) : Event
        data class DeviceDisconnected(val deviceId: String) : Event
        data class StreamingReady(val deviceId: String) : Event
        data object EcgData : Event
        data class Contact(val contact: Boolean, val supported: Boolean = true) : Event
        data class BluetoothPower(val on: Boolean) : Event
        /** Periodic timer, every few seconds. */
        data object Tick : Event
    }

    sealed interface Action {
        data class Connect(val deviceId: String) : Action
        data class Disconnect(val deviceId: String) : Action
        data class StartStreams(val deviceId: String) : Action
        data class StopStreams(val deviceId: String) : Action
        data class Log(val event: String, val detail: String = "") : Action
    }

    var state: State = State.Idle
        private set

    fun handle(event: Event, nowMs: Long): List<Action> {
        val actions = mutableListOf<Action>()
        state = next(state, event, nowMs, actions)
        return actions
    }

    private fun next(s: State, e: Event, now: Long, out: MutableList<Action>): State = when (e) {
        is Event.Start -> {
            if (s is State.Active && s !is State.Paused && s.deviceId == e.deviceId && !e.paused) {
                s
            } else {
                if (s is State.Active) teardown(s, out)
                if (e.paused) {
                    out += Action.Log("paused", "restored after restart")
                    State.Paused(e.deviceId, now)
                } else {
                    out += Action.Log("start", e.deviceId)
                    out += Action.Connect(e.deviceId)
                    State.Connecting(e.deviceId, now)
                }
            }
        }

        Event.Stop -> {
            if (s is State.Active) {
                teardown(s, out)
                out += Action.Log("stop")
            }
            State.Idle
        }

        Event.Pause -> when (s) {
            State.Idle, is State.Paused -> s
            is State.Active -> {
                teardown(s, out)
                out += Action.Log("pause")
                State.Paused(s.deviceId, now)
            }
        }

        Event.Resume -> when (s) {
            is State.Paused -> {
                out += Action.Log("resume")
                out += Action.Connect(s.deviceId)
                State.Connecting(s.deviceId, now)
            }
            else -> s
        }

        is Event.DeviceConnected -> when {
            s !is State.Active || s.deviceId != e.deviceId || s is State.Paused -> {
                // Not wanted right now: never leave a link open that nobody uses.
                out += Action.Disconnect(e.deviceId)
                s
            }
            s is State.Connecting -> {
                out += Action.Log("connected", e.deviceId)
                State.Connected(s.deviceId, now)
            }
            else -> s
        }

        is Event.StreamingReady -> when {
            s is State.Connecting && s.deviceId == e.deviceId || s is State.Connected && s.deviceId == e.deviceId -> {
                out += Action.StartStreams(e.deviceId)
                out += Action.Log("streams_start")
                State.Streaming(e.deviceId, now, now)
            }
            else -> s
        }

        is Event.DeviceDisconnected -> when {
            s is State.Active && s.deviceId == e.deviceId &&
                (s is State.Connected || s is State.Streaming || s is State.NoContact) -> {
                // Link is gone; just release the local stream collectors.
                if (s is State.Streaming) out += Action.StopStreams(s.deviceId)
                out += Action.Log("disconnected", "waiting for reconnect")
                State.Connecting(s.deviceId, now)
            }
            else -> s
        }

        Event.EcgData -> when (s) {
            is State.Streaming -> s.copy(lastDataMs = now, restarts = 0)
            else -> s
        }

        is Event.Contact -> when (s) {
            is State.Streaming -> when {
                !e.supported || e.contact -> s.copy(noContactSinceMs = null)
                s.noContactSinceMs == null -> s.copy(noContactSinceMs = now)
                else -> s
            }
            is State.NoContact -> if (e.contact || !e.supported) {
                out += Action.StartStreams(s.deviceId)
                out += Action.Log("contact_back", "streams restarted")
                State.Streaming(s.deviceId, now, now)
            } else {
                s
            }
            else -> s
        }

        is Event.BluetoothPower -> when {
            !e.on && (s is State.Connected || s is State.Streaming || s is State.NoContact) -> {
                s as State.Active
                if (s is State.Streaming) out += Action.StopStreams(s.deviceId)
                out += Action.Log("bluetooth_off")
                State.Connecting(s.deviceId, now)
            }
            e.on && s is State.Connecting -> {
                out += Action.Log("bluetooth_on")
                out += Action.Connect(s.deviceId)
                State.Connecting(s.deviceId, now, 0)
            }
            else -> s
        }

        Event.Tick -> tick(s, now, out)
    }

    private fun tick(s: State, now: Long, out: MutableList<Action>): State = when (s) {
        is State.Connecting -> if (now - s.sinceMs >= reconnect.delayMs(s.attempt)) {
            out += Action.Log("reconnect", "attempt ${s.attempt + 1}")
            out += Action.Connect(s.deviceId)
            s.copy(sinceMs = now, attempt = s.attempt + 1)
        } else {
            s
        }

        is State.Connected -> if (now - s.sinceMs >= config.noDataTimeoutMs) {
            // Streaming feature never became ready: reconnect.
            out += Action.Log("no_streaming_feature", "reconnecting")
            out += Action.Disconnect(s.deviceId)
            out += Action.Connect(s.deviceId)
            State.Connecting(s.deviceId, now)
        } else {
            s
        }

        is State.Streaming -> when {
            s.noContactSinceMs != null && now - s.noContactSinceMs >= config.noContactStopMs -> {
                out += Action.StopStreams(s.deviceId)
                out += Action.Log("no_contact", "streams stopped to save H10 battery")
                State.NoContact(s.deviceId, now)
            }
            now - s.lastDataMs >= config.noDataTimeoutMs && s.restarts < config.maxStreamRestarts -> {
                out += Action.StopStreams(s.deviceId)
                out += Action.StartStreams(s.deviceId)
                out += Action.Log("no_data", "restarting streams")
                s.copy(lastDataMs = now, restarts = s.restarts + 1)
            }
            now - s.lastDataMs >= config.noDataTimeoutMs -> {
                out += Action.StopStreams(s.deviceId)
                out += Action.Disconnect(s.deviceId)
                out += Action.Connect(s.deviceId)
                out += Action.Log("no_data", "reconnecting")
                State.Connecting(s.deviceId, now)
            }
            else -> s
        }

        else -> s
    }

    /** Leaves the device cleanly: streams first, then the link. */
    private fun teardown(s: State.Active, out: MutableList<Action>) {
        when (s) {
            is State.Streaming -> {
                out += Action.StopStreams(s.deviceId)
                out += Action.Disconnect(s.deviceId)
            }
            is State.Connecting, is State.Connected, is State.NoContact -> out += Action.Disconnect(s.deviceId)
            is State.Paused -> Unit
        }
    }
}

/** Back-off for re-issuing connect requests while the H10 is away. */
class ReconnectPolicy(
    private val delaysMs: List<Long> = listOf(30_000, 60_000, 120_000, 300_000),
) {
    init {
        require(delaysMs.isNotEmpty())
    }

    fun delayMs(attempt: Int): Long = delaysMs[attempt.coerceIn(0, delaysMs.lastIndex)]
}
