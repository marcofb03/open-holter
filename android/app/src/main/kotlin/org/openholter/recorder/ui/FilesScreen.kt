package org.openholter.recorder.ui

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.openholter.core.RecordingFiles
import org.openholter.recorder.HolterApp
import org.openholter.recorder.Storage
import java.time.LocalDate

private data class DayInfo(val day: LocalDate, val files: Int, val bytes: Long)

@Composable
fun FilesScreen() {
    val context = LocalContext.current
    val prefs = HolterApp.from(context).prefs
    val scope = rememberCoroutineScope()
    var days by remember { mutableStateOf<List<DayInfo>>(emptyList()) }
    var exportState by remember { mutableStateOf<String?>(null) }
    var refresh by remember { mutableStateOf(0) }

    LaunchedEffect(refresh) {
        days = withContext(Dispatchers.IO) {
            RecordingFiles.listDays(Storage.root(context)).map { d ->
                val files = Storage.filesOfDay(context, d)
                DayInfo(d, files.size, files.sumOf { it.length() })
            }
        }
    }

    fun export(uri: Uri) {
        scope.launch {
            exportState = "Exporting…"
            val r = withContext(Dispatchers.IO) {
                Storage.exportToTree(context, uri) { name -> exportState = "Copying $name" }
            }
            exportState = "Done: ${r.copied} copied, ${r.skipped} already there" + if (r.failed > 0) ", ${r.failed} failed" else ""
        }
    }

    val pickFolder = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) {
            context.contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
            )
            prefs.exportTreeUri = uri.toString()
            export(uri)
        }
    }

    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Recordings", style = MaterialTheme.typography.headlineSmall)
        Text(
            "Stored on this phone in ${Storage.root(context).path}. Files are removed if the app is uninstalled – export regularly.",
            style = MaterialTheme.typography.bodySmall,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            val saved = prefs.exportTreeUri
            if (saved != null) {
                Button(onClick = { export(Uri.parse(saved)) }) { Text("Export new files") }
                OutlinedButton(onClick = { pickFolder.launch(null) }) { Text("Change folder") }
            } else {
                Button(onClick = { pickFolder.launch(null) }) { Text("Export to folder…") }
            }
        }
        exportState?.let { Text(it, style = MaterialTheme.typography.bodySmall) }

        LazyColumn(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(days, key = { it.day.toString() }) { d ->
                Card(Modifier.fillMaxWidth()) {
                    Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(d.day.toString(), style = MaterialTheme.typography.titleMedium)
                            Text("${d.files} files · ${formatBytes(d.bytes)}")
                        }
                        OutlinedButton(onClick = {
                            Storage.shareDayIntent(context, d.day)?.let { context.startActivity(it) }
                        }) { Text("Share") }
                    }
                }
            }
        }
        if (days.isEmpty()) Text("No recordings yet.")
        OutlinedButton(onClick = { refresh++ }) { Text("Refresh") }
    }
}
