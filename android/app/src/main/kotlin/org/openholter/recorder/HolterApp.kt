package org.openholter.recorder

import android.app.Application
import com.polar.sdk.api.PolarBleApi
import com.polar.sdk.api.PolarBleApiDefaultImpl

class HolterApp : Application() {

    /** One Polar SDK instance per process, shared by the service and the scan screen. */
    val polar: PolarBleApi by lazy {
        PolarBleApiDefaultImpl.defaultImplementation(
            applicationContext,
            setOf(
                PolarBleApi.PolarBleSdkFeature.FEATURE_HR,
                PolarBleApi.PolarBleSdkFeature.FEATURE_BATTERY_INFO,
                PolarBleApi.PolarBleSdkFeature.FEATURE_DEVICE_INFO,
                PolarBleApi.PolarBleSdkFeature.FEATURE_POLAR_ONLINE_STREAMING,
            ),
        ).also { api ->
            api.setAutomaticReconnection(true)
            api.setApiCallback(PolarEvents.callback)
        }
    }

    val prefs: Prefs by lazy { Prefs(this) }

    companion object {
        fun from(context: android.content.Context): HolterApp = context.applicationContext as HolterApp
    }
}
