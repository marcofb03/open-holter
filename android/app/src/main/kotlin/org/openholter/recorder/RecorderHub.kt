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

    /** Last ~10 s of ECG in mV, oldest first. */
    val ecgPreview = MutableStateFlow(FloatArray(0))

    val statusFlow: StateFlow<Status> get() = status
}
