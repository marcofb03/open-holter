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
    }

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

        // Abstract in the SDK's callback base class; not used here.
        override fun disInformationReceived(identifier: String, disInfo: DisInfo) = Unit

        override fun htsNotificationReceived(identifier: String, data: PolarHealthThermometerData) = Unit
    }
}
