package org.openholter.core

/**
 * Maps the H10's sample timestamps (device clock, ns since 2000-01-01) to Unix
 * time in ns.
 *
 * The device clock gives exact spacing between samples; the phone clock gives
 * absolute time. For every packet, `arrival - lastSampleTime` is the clock
 * offset plus the Bluetooth latency. The smallest value seen in a sliding
 * window is the best offset estimate (least latency). The applied offset moves
 * towards that estimate by at most [maxSlewNs] per packet, so sample spacing
 * stays smooth while clock drift (~20 ppm) is still followed. Offsets that are
 * more than [resyncNs] off (new connection, device clock reset) are applied at
 * once.
 *
 * Mapped timestamps are strictly increasing.
 */
class ClockMapper(
    private val windowNs: Long = 120_000_000_000L,
    private val maxSlewNs: Long = 2_000_000L,
    private val resyncNs: Long = 1_000_000_000L,
) {
    private class Candidate(val arrivalNs: Long, val offsetNs: Long)

    // Monotonic deque: offsets increase from front to back, front is the window minimum.
    private val window = ArrayDeque<Candidate>()
    private var lastDeviceNs = Long.MIN_VALUE
    private var lastMappedNs = Long.MIN_VALUE

    /** Currently applied offset (Unix ns − device ns); null before the first packet. */
    var offsetNs: Long? = null
        private set

    fun reset() {
        window.clear()
        lastDeviceNs = Long.MIN_VALUE
        offsetNs = null
        // lastMappedNs is kept so that timestamps stay increasing across reconnects.
    }

    /**
     * Feed one packet: [arrivalUnixNs] is the phone time when it arrived,
     * [lastSampleDeviceNs] the device timestamp of its last sample.
     */
    fun onPacket(arrivalUnixNs: Long, lastSampleDeviceNs: Long) {
        if (lastDeviceNs != Long.MIN_VALUE && lastSampleDeviceNs < lastDeviceNs - resyncNs) reset()
        lastDeviceNs = maxOf(lastDeviceNs, lastSampleDeviceNs)

        val candidate = arrivalUnixNs - lastSampleDeviceNs
        while (window.isNotEmpty() && window.last().offsetNs >= candidate) window.removeLast()
        window.addLast(Candidate(arrivalUnixNs, candidate))
        while (window.first().arrivalNs < arrivalUnixNs - windowNs) window.removeFirst()

        val target = window.first().offsetNs
        val current = offsetNs
        offsetNs = when {
            current == null -> target
            Math.abs(target - current) > resyncNs -> target
            else -> current + (target - current).coerceIn(-maxSlewNs, maxSlewNs)
        }
    }

    /** Unix ns for a device timestamp. Call [onPacket] for the packet first. */
    fun map(deviceNs: Long): Long {
        val offset = checkNotNull(offsetNs) { "onPacket() must be called first" }
        var t = deviceNs + offset
        if (lastMappedNs != Long.MIN_VALUE && t <= lastMappedNs) t = lastMappedNs + 1
        lastMappedNs = t
        return t
    }
}
