package org.openholter.recorder

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/** Live status shared between [RecorderService] and the UI (same process). */
object RecorderHub {
    enum class Link { IDLE, CONNECTING, CONNECTED, STREAMING, NO_CONTACT, PAUSED }

    data class Status(
        val serviceRunning: Boolean = false,
        val link: Link = Link.IDLE,
        val deviceId: String? = null,
        val batteryPercent: Int? = null,
        val heartRate: Int? = null,
        val contact: Boolean? = null,
        val linkSinceMs: Long? = null,
        val todayRecordedSeconds: Long = 0,
        val lastMarkerMs: Long? = null,
        val storageUsedBytes: Long = 0,
        val storageFreeBytes: Long = 0,
        val currentFile: String? = null,
    )

    /** 10 s at 130 Hz. */
    const val PREVIEW_SAMPLES = 1300

    val status = MutableStateFlow(Status())

    /** Last ~10 s of ECG in mV, oldest first. Emptied when the ECG stream stops. */
    val ecgPreview = MutableStateFlow(FloatArray(0))

    /** Phone time (ms) of the latest ECG packet, updated about once a second; null when stopped. */
    val lastEcgMs = MutableStateFlow<Long?>(null)

    /** ECG counts as arriving if a packet came within this time (the H10 sends one every ~0.6 s). */
    const val ECG_FRESH_MS = 3_000L

    /** True only while new ECG samples are really being written. */
    fun receivingEcg(link: Link, lastEcgMs: Long?, nowMs: Long): Boolean =
        link == Link.STREAMING && lastEcgMs != null && nowMs - lastEcgMs <= ECG_FRESH_MS

    val statusFlow: StateFlow<Status> get() = status
}
