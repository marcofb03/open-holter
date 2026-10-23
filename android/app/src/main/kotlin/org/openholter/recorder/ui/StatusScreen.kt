package org.openholter.recorder.ui

import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.openholter.recorder.HolterApp
import org.openholter.recorder.RecorderHub
import org.openholter.recorder.RecorderHub.Link
import org.openholter.recorder.RecorderService
import org.openholter.recorder.Storage
import org.openholter.recorder.Updater
import java.time.LocalDate

@Composable
fun StatusScreen(
    missingPermissions: List<String>,
    onGrantPermissions: () -> Unit,
    onChooseDevice: () -> Unit,
) {
    val context = LocalContext.current
    val prefs = HolterApp.from(context).prefs
    val status by RecorderHub.statusFlow.collectAsStateWithLifecycle()
    val ecg by RecorderHub.ecgPreview.collectAsStateWithLifecycle()
    val lastEcgMs by RecorderHub.lastEcgMs.collectAsStateWithLifecycle()
    // Ticks once a second so "Recording" turns into "Starting ECG…" as soon as data stops.
    var nowMs by remember { mutableStateOf(System.currentTimeMillis()) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(1_000)
            nowMs = System.currentTimeMillis()
        }
    }
    val receivingEcg = RecorderHub.receivingEcg(status.link, lastEcgMs, nowMs)
    val update by Updater.state.collectAsStateWithLifecycle()
    val haptics = LocalHapticFeedback.current

    var recordingWanted by remember { mutableStateOf(prefs.recordingActive) }
    var batteryExempt by remember { mutableStateOf(true) }
    var storage by remember { mutableStateOf<Storage.Usage?>(null) }
    var justMarked by remember { mutableStateOf<Long?>(null) }

    LifecycleResumeEffect(Unit) {
        recordingWanted = prefs.recordingActive
        batteryExempt = context.getSystemService(PowerManager::class.java).isIgnoringBatteryOptimizations(context.packageName)
        onPauseOrDispose { }
    }
    LaunchedEffect(status.serviceRunning) {
        // When the service runs it publishes storage itself.
        while (!status.serviceRunning) {
            storage = withContext(Dispatchers.IO) { Storage.usage(context) }
            delay(30_000)
        }
    }
    LaunchedEffect(justMarked) {
        if (justMarked != null) {
            delay(4_000)
            justMarked = null
        }
    }

    val deviceId = status.deviceId ?: prefs.deviceId
    val todaySeconds = if (status.serviceRunning) status.todayRecordedSeconds else prefs.ecgSamples(LocalDate.now()) / 130
    val used = if (status.serviceRunning) status.storageUsedBytes else storage?.usedBytes ?: 0L
    val free = if (status.serviceRunning) status.storageFreeBytes else storage?.freeBytes ?: 0L
    val lastMarker = status.lastMarkerMs ?: prefs.lastMarkerMs.takeIf { it > 0 }
    val recording = status.serviceRunning && status.link != Link.IDLE

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        NotMedicalBanner()

        if (missingPermissions.isNotEmpty()) {
            InfoCard("Permissions needed") {
                Text("Bluetooth (to talk to the H10) and notifications (for the recording notification and the marker button).")
                Button(onClick = onGrantPermissions) { Text("Grant permissions") }
            }
        }
        if (!batteryExempt) {
            InfoCard("Battery optimisation is on") {
                Text("Android may stop the recorder in the background. Allow the app to run unrestricted.")
                Button(onClick = {
                    context.startActivity(
                        Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:${context.packageName}")),
                    )
                }) { Text("Allow unrestricted battery") }
            }
        }

        if (update is Updater.State.Available) {
            InfoCard("App update available") {
                Text("${(update as Updater.State.Available).release.versionName}: install it from About → App updates when it suits you.")
            }
        }

        InfoCard("Connection") {
            Text(linkText(status.link, recordingWanted, receivingEcg), style = MaterialTheme.typography.headlineSmall)
            Text("H10: ${deviceId ?: "none chosen"}" + (status.batteryPercent?.let { " · battery $it%" } ?: ""))
            status.linkSinceMs?.let { if (recording) Text("since ${formatTime(it)}", style = MaterialTheme.typography.bodySmall) }
            if (status.link == Link.NO_CONTACT) {
                Text("ECG stopped to save the H10 battery; it restarts when the strap touches the skin again. Wet the electrodes.")
            }
            Row(verticalAlignment = androidx.compose.ui.Alignment.Bottom) {
                Text(status.heartRate?.toString() ?: "–", fontSize = 40.sp, fontWeight = FontWeight.Bold)
                Text(" bpm", modifier = Modifier.padding(bottom = 8.dp))
                if (status.contact == false) Text("   no skin contact", color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(bottom = 8.dp))
            }
        }

        InfoCard("Live ECG (last 10 s)") {
            EcgTrace(if (receivingEcg) ecg else FloatArray(0), capacity = RecorderHub.PREVIEW_SAMPLES)
            if (status.link == Link.STREAMING && !receivingEcg) {
                Text("Waiting for the first ECG from the H10. Nothing is being recorded yet.", style = MaterialTheme.typography.bodySmall)
            }
        }

        Button(
            onClick = {
                haptics.performHapticFeedback(HapticFeedbackType.LongPress)
                justMarked = System.currentTimeMillis()
                RecorderService.send(context, RecorderService.ACTION_MARK)
            },
            enabled = recording,
            modifier = Modifier.fillMaxWidth().height(120.dp),
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
        ) {
            Text(
                if (justMarked != null) "Marked ${formatTime(justMarked!!)}" else "MARK SYMPTOM",
                fontSize = 28.sp, fontWeight = FontWeight.Bold,
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            when {
                deviceId == null -> Button(onClick = onChooseDevice, modifier = Modifier.weight(1f)) { Text("Choose H10") }
                !recording -> Button(
                    onClick = {
                        recordingWanted = true
                        RecorderService.send(context, RecorderService.ACTION_START, deviceId)
                    },
                    enabled = missingPermissions.isEmpty(),
                    modifier = Modifier.weight(1f),
                ) { Text("Start recording") }
                // One button. "Stop" stops the H10 streams and disconnects but keeps the
                // service and its reminder notification, and survives a reboot; "Start"
                // reconnects. (Internally this is the pause / resume of RecorderMachine.)
                status.link == Link.PAUSED -> Button(
                    onClick = { RecorderService.send(context, RecorderService.ACTION_RESUME) },
                    modifier = Modifier.weight(1f),
                ) { Text("Start recording") }
                else -> Button(
                    onClick = { RecorderService.send(context, RecorderService.ACTION_PAUSE) },
                    modifier = Modifier.weight(1f),
                ) { Text("Stop recording") }
            }
        }

        InfoCard("Recording") {
            Text("Today: %.1f h recorded".format(todaySeconds / 3600.0))
            Text("Last marker: ${lastMarker?.let { formatDateTime(it) } ?: "none"}")
            Text("Storage: ${formatBytes(used)} used · ${formatBytes(free)} free")
            status.currentFile?.let { Text("Writing $it", style = MaterialTheme.typography.bodySmall) }
        }
        Spacer(Modifier.height(8.dp))
    }
}

private fun linkText(link: Link, wanted: Boolean, receivingEcg: Boolean): String = when (link) {
    Link.IDLE -> if (wanted) "Starting…" else "Not recording"
    Link.CONNECTING -> "Searching for H10…"
    Link.CONNECTED -> "Connected, starting ECG…"
    Link.STREAMING -> if (receivingEcg) "Recording" else "Starting ECG…"
    Link.NO_CONTACT -> "No skin contact"
    Link.PAUSED -> "Stopped – tap Start recording when the strap is on"
}
