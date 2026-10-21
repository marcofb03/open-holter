package org.openholter.core

/** Fixed-size ring buffer of the most recent ECG values for the live trace. */
class EcgPreviewBuffer(val capacity: Int) {
    private val data = FloatArray(capacity)
    private var next = 0
    private var count = 0

    val size: Int get() = count

    fun add(value: Float) {
        data[next] = value
        next = (next + 1) % capacity
        if (count < capacity) count++
    }

    fun clear() {
        next = 0
        count = 0
    }

    /** Values from oldest to newest. */
    fun snapshot(): FloatArray {
        val out = FloatArray(count)
        val start = (next - count + capacity) % capacity
        for (i in 0 until count) out[i] = data[(start + i) % capacity]
        return out
    }
}
