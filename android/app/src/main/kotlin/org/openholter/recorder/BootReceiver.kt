package org.openholter.recorder

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/** Restarts recording after a reboot or an app update if it was running before. */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED && intent.action != Intent.ACTION_MY_PACKAGE_REPLACED) return
        val prefs = HolterApp.from(context).prefs
        if (!prefs.recordingActive || prefs.deviceId == null) return
        try {
            RecorderService.send(context, RecorderService.ACTION_RESTORE)
        } catch (e: Exception) {
            // E.g. Bluetooth permission revoked; the user has to open the app.
            Log.e("BootReceiver", "could not restart recording", e)
        }
    }
}
