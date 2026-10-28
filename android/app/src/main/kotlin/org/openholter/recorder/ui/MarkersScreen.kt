package org.openholter.recorder.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import org.openholter.core.HrBin
import org.openholter.core.Marker
import org.openholter.core.MarkerStore
import org.openholter.core.RecordingFiles
import org.openholter.core.SignalReaders
import org.openholter.core.Training
import org.openholter.recorder.HolterApp
import org.openholter.recorder.ReportPdf
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
 * Calendar of symptom markers (red) and trainings (blue; split cell when a day
 * has both, with the counts), the chosen day's entries with their comments,
 * and a report per marker (ECG around it, PDF) or per training (heart-rate
 * chart, summary PDF or full ECG PDF). Comments can be edited.
 */
@Composable
fun MarkersScreen() {
    val context = LocalContext.current
    val app = HolterApp.from(context)
    val zone = ZoneId.systemDefault()
    var month by remember { mutableStateOf(YearMonth.now()) }
    var markerCounts by remember { mutableStateOf<Map<LocalDate, Int>>(emptyMap()) }
    var trainingCounts by remember { mutableStateOf<Map<LocalDate, Int>>(emptyMap()) }
    var selected by remember { mutableStateOf(LocalDate.now()) }
    var dayMarkers by remember { mutableStateOf<List<Marker>>(emptyList()) }
    var dayTrainings by remember { mutableStateOf<List<Training>>(emptyList()) }
    var open by remember { mutableStateOf<Marker?>(null) }
    var openTraining by remember { mutableStateOf<Training?>(null) }
    var version by remember { mutableIntStateOf(0) } // bumped after a comment changes

    LaunchedEffect(month, version) {
        val (m, t) = withContext(Dispatchers.IO) {
            app.markers.countsByDay(month.atDay(1), month.atEndOfMonth()) to
                app.trainings.countsByDay(month.atDay(1), month.atEndOfMonth())
        }
        markerCounts = m
        trainingCounts = t
    }
    LaunchedEffect(selected, version) {
        val (m, t) = withContext(Dispatchers.IO) { app.markers.list(selected) to app.trainings.list(selected) }
        dayMarkers = m
        dayTrainings = t
    }

    val current = open
    val currentTraining = openTraining
    if (current != null) {
        MarkerReport(current, onBack = { open = null }, onChanged = { changed ->
            open = changed
            version++
        })
        return
    }
    if (currentTraining != null) {
        TrainingReport(currentTraining, onBack = { openTraining = null }, onChanged = { changed ->
            openTraining = changed
            version++
        })
        return
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        NotMedicalBanner()
        MonthCalendar(month, markerCounts, trainingCounts, selected, onMonth = { month = it }, onDay = { selected = it })
        InfoCard(LONG_DAY.format(selected)) {
            if (dayMarkers.isEmpty() && dayTrainings.isEmpty()) {
                Text("No symptom markers or trainings on this day.")
            } else {
                // Markers and trainings in time order.
                val entries = dayMarkers.map { it.timestampNs to (it as Any) } + dayTrainings.map { it.startNs to (it as Any) }
                Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 6.dp)) {
                    entries.sortedBy { it.first }.forEach { (_, e) ->
                        when (e) {
                            is Marker -> EntryCard(
                                color = MARKER_RED,
                                title = "Symptom " + CLOCK.format(localTime(e.timestampNs, zone)),
                                comment = e.comment,
                                link = "Open ECG report ›",
                            ) { open = e }
                            is Training -> EntryCard(
                                color = TRAINING_BLUE,
                                title = "Training " + CLOCK.format(localTime(e.startNs, zone)) + " – " +
                                    (e.endNs?.let { CLOCK.format(localTime(it, zone)) + " (" + ReportPdf.duration(it - e.startNs) + ")" } ?: "running"),
                                comment = e.comment,
                                link = "Open training report ›",
                            ) { openTraining = e }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun EntryCard(color: Color, title: String, comment: String, link: String, onClick: () -> Unit) {
    Card(Modifier.fillMaxWidth().clickable(onClick = onClick)) {
        Row(Modifier.height(IntrinsicSize.Min)) {
            Box(Modifier.width(6.dp).fillMaxHeight().background(color))
            Column(Modifier.padding(10.dp)) {
                Text(title, fontWeight = FontWeight.Bold, color = color)
                Text(
                    comment.ifBlank { "No comment" },
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (comment.isBlank()) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.onSurface,
                )
                Text(link, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
            }
        }
    }
}

@Composable
private fun MonthCalendar(
    month: YearMonth,
    markerCounts: Map<LocalDate, Int>,
    trainingCounts: Map<LocalDate, Int>,
    selected: LocalDate,
    onMonth: (YearMonth) -> Unit,
    onDay: (LocalDate) -> Unit,
) {
    InfoCard("Symptom markers (red) and trainings (blue)") {
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
                        DayCell(d, markerCounts[d] ?: 0, trainingCounts[d] ?: 0, d == selected, d == LocalDate.now()) { onDay(d) }
                    } else {
                        Spacer(Modifier.weight(1f))
                    }
                }
            }
        }
    }
}

/** Red for markers, blue for trainings, split half and half when a day has both; the counts in the corners. */
@Composable
private fun RowScope.DayCell(day: LocalDate, markers: Int, trainings: Int, selected: Boolean, today: Boolean, onClick: () -> Unit) {
    val shape = RoundedCornerShape(6.dp)
    val any = markers > 0 || trainings > 0
    val fg = if (any) Color.White else MaterialTheme.colorScheme.onSurfaceVariant
    var m = Modifier.weight(1f).aspectRatio(1f).padding(2.dp).clip(shape).background(MaterialTheme.colorScheme.surfaceVariant)
    if (selected) m = m.border(2.dp, MaterialTheme.colorScheme.primary, shape)
    else if (today) m = m.border(1.dp, MaterialTheme.colorScheme.outline, shape)
    Box(m.clickable(onClick = onClick)) {
        Row(Modifier.fillMaxSize()) {
            if (markers > 0) Box(Modifier.weight(1f).fillMaxHeight().background(MARKER_RED))
            if (trainings > 0) Box(Modifier.weight(1f).fillMaxHeight().background(TRAINING_BLUE))
        }
        Text("${day.dayOfMonth}", color = fg, modifier = Modifier.align(Alignment.Center), fontWeight = if (today) FontWeight.Bold else FontWeight.Normal)
        val badge = Modifier.padding(horizontal = 4.dp, vertical = 1.dp)
        if (markers > 0) {
            Text("$markers", color = fg, fontSize = 10.sp, fontWeight = FontWeight.Bold,
                modifier = Modifier.align(if (trainings > 0) Alignment.TopStart else Alignment.TopEnd).then(badge))
        }
        if (trainings > 0) {
            Text("$trainings", color = fg, fontSize = 10.sp, fontWeight = FontWeight.Bold, modifier = Modifier.align(Alignment.TopEnd).then(badge))
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
                        runCatching { ReportPdf.marker(context, marker, w, fromNs, toNs, hr) }
                    }
                    result.onSuccess {
                        exportMessage = null
                        context.startActivity(ReportPdf.shareIntent(context, it))
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

/** Bin size for a training's heart-rate chart: about 400 points or fewer, 5 s at the finest. */
private fun trainingBinNs(durationNs: Long): Long =
    listOf(5L, 10L, 15L, 30L, 60L).map { it * 1_000_000_000L }.firstOrNull { durationNs / it <= 400 } ?: 60_000_000_000L

/** Symptom markers between [fromNs] and [toNs] (they can span midnight). */
private fun markersBetween(app: HolterApp, fromNs: Long, toNs: Long, zone: ZoneId): List<Long> {
    var day = RecordingFiles.dayOf(fromNs, zone)
    val last = RecordingFiles.dayOf(toNs, zone)
    val out = mutableListOf<Long>()
    while (!day.isAfter(last)) {
        out += app.markers.list(day).map { it.timestampNs }.filter { it in fromNs..toNs }
        day = day.plusDays(1)
    }
    return out
}

@Composable
private fun TrainingReport(training: Training, onBack: () -> Unit, onChanged: (Training) -> Unit) {
    val context = LocalContext.current
    val app = HolterApp.from(context)
    val zone = ZoneId.systemDefault()
    val scope = rememberCoroutineScope()
    val start = training.startNs
    // A running training is shown up to now.
    val endNs = remember(training) { training.endNs ?: RecordingFiles.nowNs() }
    val binNs = remember(endNs) { trainingBinNs(endNs - start) }
    var bins by remember { mutableStateOf<List<HrBin>?>(null) }
    var markers by remember { mutableStateOf<List<Long>>(emptyList()) }
    var editing by remember { mutableStateOf(false) }
    var exportMessage by remember { mutableStateOf<String?>(null) }

    BackHandler(onBack = onBack)
    LaunchedEffect(start, endNs) {
        val root = Storage.root(context)
        val (b, m) = withContext(Dispatchers.IO) {
            SignalReaders.heartRate(root, zone, start, endNs, binNs) to markersBetween(app, start, endNs, zone)
        }
        bins = b
        markers = m
    }

    fun export(make: () -> java.io.File) {
        exportMessage = "Making the PDF…"
        scope.launch {
            withContext(Dispatchers.IO) { runCatching(make) }
                .onSuccess {
                    exportMessage = null
                    context.startActivity(ReportPdf.shareIntent(context, it))
                }
                .onFailure { exportMessage = "Could not make the PDF: ${it.message}" }
        }
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        TextButton(onClick = onBack) { Text("◀ Back to calendar") }
        Text("Training", style = MaterialTheme.typography.headlineSmall, color = TRAINING_BLUE)
        Text(
            "${LONG_DAY.format(localTime(start, zone))}, ${CLOCK.format(localTime(start, zone))} – " +
                CLOCK.format(localTime(endNs, zone)) + if (training.running) " (still running)" else "",
            style = MaterialTheme.typography.titleMedium,
        )
        Text("Duration: ${ReportPdf.duration(endNs - start)}")
        InfoCard("Comment") {
            Text(training.comment.ifBlank { "No comment" })
            TextButton(onClick = { editing = true }) { Text(if (training.comment.isBlank()) "Add comment" else "Edit comment") }
        }
        InfoCard("Heart rate during the training (H10)") {
            val b = bins
            when {
                b == null -> Text("Loading…")
                b.isEmpty() -> Text("No heart rate with skin contact during this training.")
                else -> {
                    HrChart(b, start, endNs, zone, binNs, markers, TRAINING_BLUE)
                    Text(ReportPdf.stats(b, binNs, start, endNs), style = MaterialTheme.typography.bodySmall)
                }
            }
        }
        Button(
            onClick = { export { ReportPdf.trainingSummary(context, training, endNs, bins.orEmpty(), binNs, markers) } },
            enabled = bins != null,
            modifier = Modifier.fillMaxWidth(),
            colors = androidx.compose.material3.ButtonDefaults.buttonColors(containerColor = TRAINING_BLUE, contentColor = Color.White),
        ) { Text("Export summary PDF (heart-rate chart, 1 page)") }
        val tooLong = endNs - start > ReportPdf.MAX_ECG_NS
        OutlinedButton(
            onClick = {
                export {
                    val w = SignalReaders.ecg(Storage.root(context), zone, start, endNs)
                    ReportPdf.trainingEcg(context, training, endNs, w, markers)
                }
            },
            enabled = !tooLong && exportMessage == null,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Export full ECG PDF (about ${ReportPdf.ecgPages(start, endNs)} pages)") }
        if (tooLong) Text("The full ECG PDF is limited to 3 hours; use the summary for longer trainings.", style = MaterialTheme.typography.bodySmall)
        exportMessage?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        Text(
            "Heart rate as reported by the H10 strap, only times with skin contact; red lines are symptom markers. " +
                "The full ECG PDF is on standard ECG paper (25 mm/s, 10 mm/mV). Not a medical device.",
            style = MaterialTheme.typography.bodySmall,
        )
    }

    if (editing) {
        CommentDialog(
            title = "Comment for the training at ${CLOCK.format(localTime(start, zone))}",
            initial = training.comment,
            confirmLabel = "Save",
            dismissLabel = "Cancel",
            placeholder = "e.g. tennis, stairs, meditation",
            onSave = { text ->
                editing = false
                scope.launch {
                    withContext(Dispatchers.IO) { app.trainings.setComment(start, text) }
                    onChanged(training.copy(comment = MarkerStore.clean(text)))
                }
            },
            onDismiss = { editing = false },
        )
    }
}
