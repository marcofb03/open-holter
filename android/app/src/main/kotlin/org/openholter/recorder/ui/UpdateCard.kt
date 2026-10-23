package org.openholter.recorder.ui

import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.launch
import org.openholter.recorder.BuildConfig
import org.openholter.recorder.Updater
import org.openholter.recorder.Updater.State

/** "Check for updates" / "Update now". Nothing is installed unless the user taps Update now. */
@Composable
fun UpdateCard() {
    val context = LocalContext.current
    val state by Updater.state.collectAsStateWithLifecycle()
    var canInstall by remember { mutableStateOf(context.packageManager.canRequestPackageInstalls()) }
    LifecycleResumeEffect(Unit) {
        canInstall = context.packageManager.canRequestPackageInstalls()
        onPauseOrDispose { }
    }
    val busy = state is State.Checking || state is State.Downloading || state is State.Installing

    InfoCard("App updates") {
        Text("Installed: ${BuildConfig.VERSION_NAME} (build ${BuildConfig.VERSION_CODE})")
        Text(
            when (val s = state) {
                State.Idle -> "Updates come from the newest build of the project on GitHub."
                State.Checking -> "Checking…"
                is State.UpToDate -> "Up to date (newest: ${s.latest.versionName}, build ${s.latest.versionCode})."
                is State.Available -> "Update available: ${s.release.versionName} (build ${s.release.versionCode})."
                is State.Downloading -> "Downloading ${s.release.versionName}… ${s.percent}%"
                is State.Installing -> "Installing ${s.release.versionName}. The app closes and restarts by itself."
                is State.Failed -> s.message
            },
            color = if (state is State.Failed) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface,
        )
        if (!canInstall) {
            Text("To install updates from here, allow this app once under \"Install unknown apps\".")
            OutlinedButton(onClick = {
                context.startActivity(
                    Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${context.packageName}")),
                )
            }) { Text("Allow app updates") }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { Updater.scope.launch { Updater.check() } }, enabled = !busy) {
                Text("Check for updates")
            }
            val s = state
            if (s is State.Available) {
                Button(
                    onClick = { Updater.scope.launch { Updater.downloadAndInstall(context, s.release) } },
                    enabled = canInstall,
                ) { Text("Update now") }
            }
        }
        Text(
            "While recording, the ECG stops for about half a minute during the update and then " +
                "starts again by itself. The event log shows it (update_start, update_installed).",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
