package org.openholter.recorder

import com.polar.androidcommunications.api.ble.model.DisInfo
import com.polar.sdk.api.PolarBleApi
import com.polar.sdk.api.PolarBleApiCallback
import com.polar.sdk.api.PolarBleDisconnectInfo
import com.polar.sdk.api.model.PolarDeviceInfo
import com.polar.sdk.api.model.PolarHealthThermometerData
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.util.UUID

/** Polar SDK callbacks turned into a flow, collected by [RecorderService]. */
object PolarEvents {
    sealed interface E {
        data class Connecting(val deviceId: String) : E
        data class Connected(val deviceId: String) : E
        data class Disconnected(val deviceId: String, val reason: String) : E
        data class StreamingReady(val deviceId: String) : E
        data class HrReady(val deviceId: String) : E
        data class Battery(val deviceId: String, val percent: Int) : E
        data class Power(val on: Boolean) : E
        /** Device Information Service string, e.g. "firmware" → "3.1.1". */
        data class Info(val deviceId: String, val name: String, val value: String) : E
    }

    private val DIS_NAMES = mapOf(
        UUID.fromString("00002a24-0000-1000-8000-00805f9b34fb") to "model",
        UUID.fromString("00002a26-0000-1000-8000-00805f9b34fb") to "firmware",
        UUID.fromString("00002a27-0000-1000-8000-00805f9b34fb") to "hardware",
        UUID.fromString("00002a28-0000-1000-8000-00805f9b34fb") to "software",
    )

    private val _events = MutableSharedFlow<E>(extraBufferCapacity = 256, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    val events: SharedFlow<E> = _events

    val callback = object : PolarBleApiCallback() {
        override fun blePowerStateChanged(powered: Boolean) {
            _events.tryEmit(E.Power(powered))
        }

        override fun deviceConnecting(polarDeviceInfo: PolarDeviceInfo) {
            _events.tryEmit(E.Connecting(polarDeviceInfo.deviceId))
        }

        override fun deviceConnected(polarDeviceInfo: PolarDeviceInfo) {
            _events.tryEmit(E.Connected(polarDeviceInfo.deviceId))
        }

        // The SDK calls this and the deprecated single-argument overload; only this one is used.
        override fun deviceDisconnected(polarDeviceInfo: PolarDeviceInfo, info: PolarBleDisconnectInfo) {
            _events.tryEmit(E.Disconnected(polarDeviceInfo.deviceId, info.reason.name))
        }

        override fun bleSdkFeatureReady(identifier: String, feature: PolarBleApi.PolarBleSdkFeature) {
            when (feature) {
                PolarBleApi.PolarBleSdkFeature.FEATURE_POLAR_ONLINE_STREAMING -> _events.tryEmit(E.StreamingReady(identifier))
                PolarBleApi.PolarBleSdkFeature.FEATURE_HR -> _events.tryEmit(E.HrReady(identifier))
                else -> Unit
            }
        }

        override fun batteryLevelReceived(identifier: String, level: Int) {
            _events.tryEmit(E.Battery(identifier, level))
        }

        // Standard Bluetooth Device Information Service strings. ECG streaming needs
        // H10 firmware 3.0.35 or later, so the firmware goes into the event log.
        override fun disInformationReceived(identifier: String, uuid: UUID, value: String) {
            DIS_NAMES[uuid]?.let { _events.tryEmit(E.Info(identifier, it, value.trim())) }
        }

        // Abstract in the SDK's callback base class; the UUID overload above is used.
        override fun disInformationReceived(identifier: String, disInfo: DisInfo) = Unit

        override fun htsNotificationReceived(identifier: String, data: PolarHealthThermometerData) = Unit
    }
}
