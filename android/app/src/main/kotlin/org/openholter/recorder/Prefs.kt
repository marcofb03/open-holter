package org.openholter.recorder

import android.content.Context
import androidx.core.content.edit
import java.time.LocalDate

/** Small persistent settings and counters. */
class Prefs(context: Context) {
    private val sp = context.getSharedPreferences("recorder", Context.MODE_PRIVATE)

    var deviceId: String?
        get() = sp.getString("device_id", null)
        set(v) = sp.edit { putString("device_id", v) }

    /** True while the user wants recording (also while paused); used to restart after reboot. */
    var recordingActive: Boolean
        get() = sp.getBoolean("recording_active", false)
        set(v) = sp.edit { putBoolean("recording_active", v) }

    var paused: Boolean
        get() = sp.getBoolean("paused", false)
        set(v) = sp.edit { putBoolean("paused", v) }

    /** Accelerometer sample rate (25 or 50 Hz). */
    var accRateHz: Int
        get() = sp.getInt("acc_rate_hz", 25)
        set(v) = sp.edit { putInt("acc_rate_hz", v) }

    var lastMarkerMs: Long
        get() = sp.getLong("last_marker_ms", 0L)
        set(v) = sp.edit { putLong("last_marker_ms", v) }

    /** Version that started an app update while recording; logged by the new version. */
    var updateFrom: String?
        get() = sp.getString("update_from", null)
        set(v) = sp.edit { putString("update_from", v) }

    var exportTreeUri: String?
        get() = sp.getString("export_tree_uri", null)
        set(v) = sp.edit { putString("export_tree_uri", v) }

    fun ecgSamples(day: LocalDate): Long = sp.getLong("samples_$day", 0L)

    fun setEcgSamples(day: LocalDate, count: Long) = sp.edit { putLong("samples_$day", count) }
}
