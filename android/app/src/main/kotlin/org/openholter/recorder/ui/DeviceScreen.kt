package org.openholter.recorder.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.polar.sdk.api.model.PolarDeviceInfo
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.withTimeoutOrNull
import org.openholter.recorder.HolterApp
import org.openholter.recorder.RecorderHub
import org.openholter.recorder.RecorderService

@Composable
fun DeviceScreen(
    missingPermissions: List<String>,
    onGrantPermissions: () -> Unit,
    onStarted: () -> Unit,
) {
    val context = LocalContext.current
    val app = HolterApp.from(context)
    val prefs = app.prefs
    val status by RecorderHub.statusFlow.collectAsStateWithLifecycle()

    var saved by remember { mutableStateOf(prefs.deviceId) }
    var accRate by remember { mutableStateOf(prefs.accRateHz) }
    var scanning by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    val found = remember { mutableStateListOf<PolarDeviceInfo>() }

    LaunchedEffect(scanning) {
        if (!scanning) return@LaunchedEffect
        found.clear()
        error = null
        withTimeoutOrNull(30_000) {
            app.polar.searchForDevice()
                .catch { error = it.message ?: it.toString() }
                .collect { info ->
                    if (found.none { it.deviceId == info.deviceId }) found += info
                }
        }
        scanning = false
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Polar H10", style = MaterialTheme.typography.headlineSmall)
        InfoCard("Remembered strap") {
            Text(saved ?: "none – scan below and tap your H10")
            if (saved != null) {
                OutlinedButton(
                    enabled = !(status.serviceRunning && status.link != RecorderHub.Link.IDLE),
                    onClick = {
                        prefs.deviceId = null
                        saved = null
                    },
                ) { Text("Forget") }
            }
        }

        if (missingPermissions.isNotEmpty()) {
            Button(onClick = onGrantPermissions) { Text("Grant Bluetooth permissions") }
        } else {
            Text("Put the strap on (wet the electrodes) so the H10 wakes up, then scan.")
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = { scanning = !scanning }) { Text(if (scanning) "Stop scan" else "Scan") }
                if (scanning) CircularProgressIndicator()
            }
        }
        error?.let { Text("Scan failed: $it", color = MaterialTheme.colorScheme.error) }

        found.sortedByDescending { it.rssi }.forEach { info ->
            Card(
                Modifier.fillMaxWidth().clickable {
                    scanning = false
                    prefs.deviceId = info.deviceId
                    saved = info.deviceId
                    RecorderService.send(context, RecorderService.ACTION_START, info.deviceId)
                    onStarted()
                },
            ) {
                Column(Modifier.padding(12.dp)) {
                    Text(info.name.ifBlank { "Polar device" }, style = MaterialTheme.typography.titleMedium)
                    Text("ID ${info.deviceId} · signal ${info.rssi} dBm")
                    if (!info.name.contains("H10")) Text("Not an H10 – ECG may not be available", color = MaterialTheme.colorScheme.error)
                }
            }
        }

        InfoCard("Accelerometer rate") {
            Text("Recorded together with the ECG to recognise movement artefacts. Takes effect at the next (re)connect.")
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(25, 50).forEach { hz ->
                    FilterChip(
                        selected = accRate == hz,
                        onClick = {
                            accRate = hz
                            prefs.accRateHz = hz
                        },
                        label = { Text("$hz Hz") },
                    )
                }
            }
        }
    }
}
