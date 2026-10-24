package org.openholter.recorder

import android.app.Application
import com.polar.sdk.api.PolarBleApi
import com.polar.sdk.api.PolarBleApiDefaultImpl
import org.openholter.core.MarkerStore
import java.time.ZoneId

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

    /** Symptom markers and their comments, shared by the service, the receiver and the UI. */
    val markers: MarkerStore by lazy { MarkerStore(Storage.root(this), ZoneId.systemDefault()) }

    companion object {
        fun from(context: android.content.Context): HolterApp = context.applicationContext as HolterApp
    }
}
