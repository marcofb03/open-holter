package org.openholter.recorder.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import org.openholter.recorder.R
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

@Composable
fun NotMedicalBanner(modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer,
            contentColor = MaterialTheme.colorScheme.onErrorContainer,
        ),
    ) {
        Text(
            stringResource(R.string.not_medical),
            modifier = Modifier.padding(12.dp),
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Medium,
        )
    }
}

@Composable
fun InfoCard(title: String, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Card(modifier = modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall)
            content()
        }
    }
}

fun formatBytes(bytes: Long): String = when {
    bytes >= 1L shl 30 -> "%.1f GB".format(bytes / (1L shl 30).toDouble())
    bytes >= 1L shl 20 -> "%.0f MB".format(bytes / (1L shl 20).toDouble())
    else -> "%.0f kB".format(bytes / 1024.0)
}

private val TIME = DateTimeFormatter.ofPattern("HH:mm:ss")
private val DATE_TIME = DateTimeFormatter.ofPattern("EEE d MMM, HH:mm:ss")

fun formatTime(ms: Long): String = TIME.format(Instant.ofEpochMilli(ms).atZone(ZoneId.systemDefault()))

fun formatDateTime(ms: Long): String = DATE_TIME.format(Instant.ofEpochMilli(ms).atZone(ZoneId.systemDefault()))

/** Colour of trainings (button, calendar, chart), next to the red of symptom markers. */
val TRAINING_BLUE = androidx.compose.ui.graphics.Color(0xFF1565C0)

/** "1:05:09" or "5:09". */
fun formatElapsed(ms: Long): String {
    val s = (ms / 1000).coerceAtLeast(0)
    return if (s >= 3600) "%d:%02d:%02d".format(s / 3600, s / 60 % 60, s % 60) else "%d:%02d".format(s / 60, s % 60)
}

/**
 * Optional comment for a symptom marker or a training. [confirmLabel] saves (also an empty
 * comment); [dismissLabel] closes without changing anything.
 */
@Composable
fun CommentDialog(
    title: String,
    initial: String,
    confirmLabel: String,
    dismissLabel: String,
    onSave: (String) -> Unit,
    onDismiss: () -> Unit,
    placeholder: String = "e.g. palpitations climbing stairs",
) {
    var text by remember { mutableStateOf(initial) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            OutlinedTextField(
                value = text,
                onValueChange = { text = it.take(500) },
                placeholder = { Text(placeholder) },
                label = { Text("Comment (optional)") },
                minLines = 2,
                modifier = Modifier.fillMaxWidth(),
            )
        },
        confirmButton = { TextButton(onClick = { onSave(text) }) { Text(confirmLabel) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text(dismissLabel) } },
    )
}
