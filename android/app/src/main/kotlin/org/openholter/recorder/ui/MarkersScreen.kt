package org.openholter.recorder.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.openholter.core.EcgWindow
import org.openholter.core.Marker
import org.openholter.core.MarkerStore
import org.openholter.core.RecordingFiles
import org.openholter.core.SignalReaders
import org.openholter.recorder.HolterApp
import org.openholter.recorder.MarkerPdf
import org.openholter.recorder.Storage
import java.time.LocalDate
import java.time.YearMonth
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.ceil

private val MARKER_RED = Color(0xFFD32F2F)
private val MONTH_FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("MMMM yyyy")
private val LONG_DAY: DateTimeFormatter = DateTimeFormatter.ofPattern("EEE d MMM yyyy")
private val CLOCK: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss")

/** Seconds of ECG after the marker in the report. */
const val MARKER_AFTER_S = 15

private fun localTime(ns: Long, zone: ZoneId) = RecordingFiles.instantOf(ns).atZone(zone)

/**
 * Calendar of symptom markers (red days with the number of markers), the
 * markers of the chosen day with their comments, and a report per marker with
 * the raw ECG around it, an editable comment and a PDF export.
 */
@Composable
fun MarkersScreen() {
    val context = LocalContext.current
    val store = HolterApp.from(context).markers
    val zone = ZoneId.systemDefault()
    var month by remember { mutableStateOf(YearMonth.now()) }
    var counts by remember { mutableStateOf<Map<LocalDate, Int>>(emptyMap()) }
    var selected by remember { mutableStateOf(LocalDate.now()) }
    var dayMarkers by remember { mutableStateOf<List<Marker>>(emptyList()) }
    var open by remember { mutableStateOf<Marker?>(null) }
    var version by remember { mutableIntStateOf(0) } // bumped after a comment changes

    LaunchedEffect(month, version) {
        counts = withContext(Dispatchers.IO) { store.countsByDay(month.atDay(1), month.atEndOfMonth()) }
    }
    LaunchedEffect(selected, version) {
        dayMarkers = withContext(Dispatchers.IO) { store.list(selected) }
    }

    val current = open
    if (current != null) {
        MarkerReport(current, onBack = { open = null }, onChanged = { changed ->
            open = changed
            version++
        })
        return
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        NotMedicalBanner()
        MonthCalendar(month, counts, selected, onMonth = { month = it }, onDay = { selected = it })
        InfoCard("Markers on ${LONG_DAY.format(selected)}") {
            if (dayMarkers.isEmpty()) {
                Text("No symptom markers on this day.")
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 6.dp)) {
                    dayMarkers.forEach { m ->
                        Card(Modifier.fillMaxWidth().clickable { open = m }) {
                            Column(Modifier.padding(10.dp)) {
                                Text(CLOCK.format(localTime(m.timestampNs, zone)), fontWeight = FontWeight.Bold)
                                Text(
                                    m.comment.ifBlank { "No comment" },
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = if (m.comment.isBlank()) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.onSurface,
                                )
                                Text("Open ECG report ›", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun MonthCalendar(
    month: YearMonth,
    counts: Map<LocalDate, Int>,
    selected: LocalDate,
    onMonth: (YearMonth) -> Unit,
    onDay: (LocalDate) -> Unit,
) {
    InfoCard("Symptom markers") {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = { onMonth(month.minusMonths(1)) }) { Text("◀") }
            Text(MONTH_FORMAT.format(month), modifier = Modifier.weight(1f), textAlign = TextAlign.Center, style = MaterialTheme.typography.titleMedium)
            TextButton(onClick = { onMonth(month.plusMonths(1)) }, enabled = month < YearMonth.now()) { Text("▶") }
        }
        Row {
            listOf("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun").forEach {
                Text(it, modifier = Modifier.weight(1f), textAlign = TextAlign.Center, style = MaterialTheme.typography.labelSmall)
            }
        }
        val offset = month.atDay(1).dayOfWeek.value - 1 // Monday first
        val days = month.lengthOfMonth()
        val rows = ceil((offset + days) / 7.0).toInt()
        for (r in 0 until rows) {
            Row {
                for (c in 0 until 7) {
                    val n = r * 7 + c - offset
                    if (n in 0 until days) {
                        val d = month.atDay(n + 1)
                        DayCell(d, counts[d] ?: 0, d == selected, d == LocalDate.now()) { onDay(d) }
                    } else {
                        Spacer(Modifier.weight(1f))
                    }
                }
            }
        }
    }
}

@Composable
private fun RowScope.DayCell(day: LocalDate, count: Int, selected: Boolean, today: Boolean, onClick: () -> Unit) {
    val shape = RoundedCornerShape(6.dp)
    val bg = if (count > 0) MARKER_RED else MaterialTheme.colorScheme.surfaceVariant
    val fg = if (count > 0) Color.White else MaterialTheme.colorScheme.onSurfaceVariant
    var m = Modifier.weight(1f).aspectRatio(1f).padding(2.dp).clip(shape).background(bg)
    if (selected) m = m.border(2.dp, MaterialTheme.colorScheme.primary, shape)
    else if (today) m = m.border(1.dp, MaterialTheme.colorScheme.outline, shape)
    Box(m.clickable(onClick = onClick)) {
        Text("${day.dayOfMonth}", color = fg, modifier = Modifier.align(Alignment.Center), fontWeight = if (today) FontWeight.Bold else FontWeight.Normal)
        if (count > 0) {
            Text("$count", color = fg, fontSize = 10.sp, fontWeight = FontWeight.Bold, modifier = Modifier.align(Alignment.TopEnd).padding(end = 4.dp, top = 1.dp))
        }
    }
}

@Composable
private fun MarkerReport(marker: Marker, onBack: () -> Unit, onChanged: (Marker) -> Unit) {
    val context = LocalContext.current
    val app = HolterApp.from(context)
    val zone = ZoneId.systemDefault()
    val scope = rememberCoroutineScope()
    var beforeS by remember { mutableIntStateOf(60) }
    var window by remember { mutableStateOf<EcgWindow?>(null) }
    var hr by remember { mutableStateOf<Int?>(null) }
    var editing by remember { mutableStateOf(false) }
    var exportMessage by remember { mutableStateOf<String?>(null) }
    val ts = marker.timestampNs
    val fromNs = ts - beforeS * 1_000_000_000L
    val toNs = ts + MARKER_AFTER_S * 1_000_000_000L

    BackHandler(onBack = onBack)
    LaunchedEffect(ts, beforeS) {
        window = null
        val root = Storage.root(context)
        window = withContext(Dispatchers.IO) { SignalReaders.ecg(root, zone, fromNs, toNs) }
        hr = withContext(Dispatchers.IO) {
            SignalReaders.heartRate(root, zone, ts - 10_000_000_000L, ts + 10_000_000_000L, binNs = 20_000_000_000L)
                .firstOrNull()?.median
        }
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        TextButton(onClick = onBack) { Text("◀ Back to calendar") }
        val at = localTime(ts, zone)
        Text("Symptom marker", style = MaterialTheme.typography.headlineSmall)
        Text("${LONG_DAY.format(at)}, ${CLOCK.format(at)}", style = MaterialTheme.typography.titleMedium)
        InfoCard("Comment") {
            Text(marker.comment.ifBlank { "No comment" })
            TextButton(onClick = { editing = true }) { Text(if (marker.comment.isBlank()) "Add comment" else "Edit comment") }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            FilterChip(selected = beforeS == 30, onClick = { beforeS = 30 }, label = { Text("30 s before") })
            FilterChip(selected = beforeS == 60, onClick = { beforeS = 60 }, label = { Text("60 s before") })
            Text("+ $MARKER_AFTER_S s after", style = MaterialTheme.typography.bodySmall)
        }
        InfoCard("ECG around the marker") {
            val w = window
            when {
                w == null -> Text("Loading…")
                w.size == 0 -> Text("No ECG was recorded around this marker (the H10 was not connected, or recording was stopped).")
                else -> {
                    val expected = (toNs - fromNs) / 1e9 * 130
                    Text(
                        "%.0f %% of the time has ECG".format((100.0 * w.size / expected).coerceAtMost(100.0)) +
                            (hr?.let { " · H10 heart rate around the marker: $it bpm" } ?: ""),
                        style = MaterialTheme.typography.bodySmall,
                    )
                    EcgStrip(w, fromNs, toNs, ts, zone)
                }
            }
        }
        Button(
            onClick = {
                val w = window ?: return@Button
                exportMessage = "Making the PDF…"
                scope.launch {
                    val result = withContext(Dispatchers.IO) {
                        runCatching { MarkerPdf.write(context, marker, w, fromNs, toNs, hr) }
                    }
                    result.onSuccess {
                        exportMessage = null
                        context.startActivity(MarkerPdf.shareIntent(context, it))
                    }.onFailure { exportMessage = "Could not make the PDF: ${it.message}" }
                }
            },
            enabled = window != null,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Export PDF (for a doctor)") }
        exportMessage?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        Text(
            "Signal of the H10 chest strap as recorded (130 Hz, no extra filtering by the app). Grid: 0.2 s and 0.5 mV. " +
                "The PDF uses standard ECG paper scale (25 mm/s, 10 mm/mV). " +
                "Not a medical device; a doctor must read it.",
            style = MaterialTheme.typography.bodySmall,
        )
    }

    if (editing) {
        CommentDialog(
            title = "Comment for ${CLOCK.format(localTime(ts, zone))}",
            initial = marker.comment,
            confirmLabel = "Save",
            dismissLabel = "Cancel",
            onSave = { text ->
                editing = false
                scope.launch {
                    withContext(Dispatchers.IO) { app.markers.setComment(ts, text) }
                    onChanged(marker.copy(comment = MarkerStore.clean(text)))
                }
            },
            onDismiss = { editing = false },
        )
    }
}

/**
 * The ECG of [fromNs]..[toNs] in rows of [secondsPerRow] s, ECG-paper style,
 * with the marker as a red line. The same vertical scale in every row
 * (3.5 mV per row, baseline at 60 % of the height).
 */
@Composable
private fun EcgStrip(w: EcgWindow, fromNs: Long, toNs: Long, markerNs: Long, zone: ZoneId, secondsPerRow: Int = 5) {
    val rowNs = secondsPerRow * 1_000_000_000L
    val rows = ceil((toNs - fromNs).toDouble() / rowNs).toInt()
    val minor = Color(0xFFF6C9C9)
    val major = Color(0xFFE59A9A)
    val ink = Color.Black // on white paper, also in dark mode
    for (r in 0 until rows) {
        val a = fromNs + r * rowNs
        val b = a + rowNs
        val i0 = w.t.binarySearch(a).let { if (it < 0) -it - 1 else it }
        val i1 = w.t.binarySearch(b).let { if (it < 0) -it - 1 else it }
        Text(CLOCK.format(localTime(a, zone)), style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(top = 4.dp))
        Canvas(Modifier.fillMaxWidth().height(110.dp).background(Color.White)) {
            val width = size.width
            val height = size.height
            val pxPerS = width / secondsPerRow
            val pxPerMv = height / 3.5f
            var s = 0f
            var k = 0
            while (s <= secondsPerRow + 1e-3f) {
                val x = s * pxPerS
                drawLine(if (k % 5 == 0) major else minor, Offset(x, 0f), Offset(x, height), strokeWidth = if (k % 5 == 0) 1.5f else 1f)
                s += 0.2f
                k++
            }
            val baseY = height * 0.6f
            var mv = -2f
            while (mv <= 2.5f) {
                val y = baseY - mv * pxPerMv
                if (y in 0f..height) drawLine(major, Offset(0f, y), Offset(width, y), strokeWidth = 1f)
                mv += 0.5f
            }
            if (i1 > i0) {
                val slice = w.mv.copyOfRange(i0, i1).sortedArray()
                val median = slice[slice.size / 2]
                val path = Path()
                var prevT = Long.MIN_VALUE
                for (i in i0 until i1) {
                    val x = (w.t[i] - a) / 1e9f * pxPerS
                    val y = baseY - (w.mv[i] - median) * pxPerMv
                    // A gap of more than 50 ms is not bridged.
                    if (prevT == Long.MIN_VALUE || w.t[i] - prevT > 50_000_000L) path.moveTo(x, y) else path.lineTo(x, y)
                    prevT = w.t[i]
                }
                clipRect { drawPath(path, ink, style = Stroke(width = 1.6f)) }
            }
            if (markerNs in a until b) {
                val x = (markerNs - a) / 1e9f * pxPerS
                drawLine(MARKER_RED, Offset(x, 0f), Offset(x, height), strokeWidth = 3f)
            }
        }
    }
}
