package org.openholter.recorder.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.openholter.core.EcgSnap
import org.openholter.core.EcgWindow
import org.openholter.core.Marker
import org.openholter.core.Measurement
import org.openholter.core.RecordingFiles
import org.openholter.core.SignalReaders
import org.openholter.recorder.HolterApp
import org.openholter.recorder.ReportPdf
import org.openholter.recorder.Storage
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.ceil

private val MEASURE_BLUE = Color(0xFF1565C0)
private val MS_CLOCK: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss.SSS")
private const val ROW_NS = 10_000_000_000L

/**
 * Measure intervals on the ECG of a marker: the same 10-s rows as the PDF (normal or
 * shifted by 5 s). Tap two points; each one snaps to the tip of the beat within 50 ms
 * (or stays where tapped in "free" mode) and a blue arrow shows the time between them in
 * ms and as a rate in bpm. Measurements are saved with the marker and go into
 * "Export PDF with my measurements".
 */
@Composable
fun PersonalizeScreen(marker: Marker, fromNs: Long, toNs: Long, hr: Int?, onBack: () -> Unit) {
    val context = LocalContext.current
    val app = HolterApp.from(context)
    val zone = ZoneId.systemDefault()
    val scope = rememberCoroutineScope()
    var window by remember { mutableStateOf<EcgWindow?>(null) }
    var measurements by remember { mutableStateOf<List<Measurement>>(emptyList()) }
    var snapToPeak by remember { mutableStateOf(true) }
    var shifted by remember { mutableStateOf(false) }
    var zoom by remember { mutableIntStateOf(1) }
    var pending by remember { mutableStateOf<Long?>(null) }
    var exportMessage by remember { mutableStateOf<String?>(null) }
    val ts = marker.timestampNs

    BackHandler(onBack = onBack)
    LaunchedEffect(ts, fromNs, toNs) {
        val (w, m) = withContext(Dispatchers.IO) {
            SignalReaders.ecg(Storage.root(context), zone, fromNs - ReportPdf.HALF_ROW_NS, toNs + ReportPdf.HALF_ROW_NS) to
                app.measurements.list(ts)
        }
        window = w
        measurements = m
    }

    val rowsFrom = if (shifted) fromNs - ReportPdf.HALF_ROW_NS else fromNs
    val rowsTo = if (shifted) toNs + ReportPdf.HALF_ROW_NS else toNs
    val rows = ceil((rowsTo - rowsFrom).toDouble() / ROW_NS).toInt()

    fun tap(tNs: Long) {
        val w = window ?: return
        val point = (if (snapToPeak) EcgSnap.peak(w, tNs) else EcgSnap.nearest(w, tNs)) ?: return
        val first = pending
        if (first == null) {
            pending = point
        } else {
            pending = null
            if (point != first) {
                val m = Measurement(minOf(first, point), maxOf(first, point))
                measurements = (measurements + m).sortedBy { it.aNs }
                scope.launch { withContext(Dispatchers.IO) { app.measurements.add(ts, m) } }
            }
        }
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        TextButton(onClick = onBack) { Text("◀ Back to the marker") }
        Text("Measure intervals", style = MaterialTheme.typography.headlineSmall)
        Text(
            if (pending == null) "Tap a point on the ECG, then a second one." else "Now tap the second point.",
            style = MaterialTheme.typography.bodyMedium,
            color = if (pending == null) MaterialTheme.colorScheme.onSurface else MEASURE_BLUE,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            FilterChip(selected = snapToPeak, onClick = { snapToPeak = true }, label = { Text("Snap to peak") })
            FilterChip(selected = !snapToPeak, onClick = { snapToPeak = false }, label = { Text("Free") })
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            FilterChip(selected = !shifted, onClick = { shifted = false }, label = { Text("Rows as PDF") })
            FilterChip(selected = shifted, onClick = { shifted = true }, label = { Text("Shifted 5 s") })
            listOf(1, 2, 4).forEach { z -> FilterChip(selected = zoom == z, onClick = { zoom = z }, label = { Text("${z}×") }) }
        }

        val w = window
        if (w == null) {
            Text("Loading…")
        } else if (w.size == 0) {
            Text("No ECG was recorded around this marker.")
        } else {
            val measurer = rememberTextMeasurer()
            BoxWithConstraints(Modifier.fillMaxWidth()) {
                val rowWidth = maxWidth * zoom
                Box(Modifier.horizontalScroll(rememberScrollState())) {
                    Column(Modifier.width(rowWidth)) {
                        for (r in 0 until rows) {
                            val rowStart = rowsFrom + r * ROW_NS
                            Text(
                                CLOCK_S.format(RecordingFiles.instantOf(rowStart).atZone(zone)),
                                style = MaterialTheme.typography.labelSmall,
                                modifier = Modifier.padding(top = 6.dp),
                            )
                            Canvas(
                                Modifier.fillMaxWidth().height(140.dp).background(Color.White)
                                    .pointerInput(rowStart, snapToPeak, pending, window, shifted) {
                                        detectTapGestures { pos -> tap(rowStart + (pos.x / size.width * ROW_NS).toLong()) }
                                    },
                            ) {
                                drawEcgRow(w, rowStart, ts, measurements, pending, measurer)
                            }
                        }
                    }
                }
            }
        }

        if (measurements.isNotEmpty()) {
            InfoCard("Measurements") {
                measurements.forEachIndexed { i, m ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text("${i + 1}.  %.0f ms · %.0f bpm".format(m.intervalMs, m.bpm), color = MEASURE_BLUE)
                            Text(
                                MS_CLOCK.format(RecordingFiles.instantOf(m.aNs).atZone(zone)) + " → " +
                                    MS_CLOCK.format(RecordingFiles.instantOf(m.bNs).atZone(zone)),
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                        TextButton(onClick = {
                            measurements = measurements - m
                            scope.launch { withContext(Dispatchers.IO) { app.measurements.remove(ts, m) } }
                        }) { Text("Delete") }
                    }
                }
            }
        }

        Button(
            onClick = {
                val win = window ?: return@Button
                exportMessage = "Making the PDF…"
                scope.launch {
                    withContext(Dispatchers.IO) {
                        runCatching { ReportPdf.marker(context, marker, win, rowsFrom, rowsTo, hr, shifted, measurements) }
                    }.onSuccess {
                        exportMessage = null
                        context.startActivity(ReportPdf.shareIntent(context, it))
                    }.onFailure { exportMessage = "Could not make the PDF: ${it.message}" }
                }
            },
            enabled = window != null && exportMessage == null,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Export PDF with my measurements") }
        exportMessage?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        Text(
            "\"Snap to peak\" puts each point on the highest point of the signal within 50 ms of your tap (the tip " +
                "of a beat), good for the time from beat to beat. \"Free\" keeps the point where you tap, e.g. for the " +
                "start and end of one beat; zoom in for that. The H10 records every 7.7 ms, so a measurement is " +
                "accurate to about ±8 ms. bpm = 60 000 ÷ ms. Not a medical device.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

private val CLOCK_S: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss")

/** One 10-s row: ECG-paper grid, the signal (3.5 mV per row, baseline at 60 %), the marker, measurements. */
private fun DrawScope.drawEcgRow(
    w: EcgWindow, rowStart: Long, markerNs: Long, measurements: List<Measurement>, pending: Long?,
    measurer: androidx.compose.ui.text.TextMeasurer,
) {
    val rowEnd = rowStart + ROW_NS
    val width = size.width
    val height = size.height
    val pxPerS = width / 10f
    val pxPerMv = height / 3.5f
    val minor = Color(0xFFF6C9C9)
    val major = Color(0xFFE59A9A)
    var k = 0
    while (k <= 50) { // 0.2 s lines, darker every second
        val x = k * 0.2f * pxPerS
        drawLine(if (k % 5 == 0) major else minor, Offset(x, 0f), Offset(x, height), strokeWidth = if (k % 5 == 0) 1.5f else 1f)
        k++
    }
    val baseY = height * 0.6f
    var mv = -2f
    while (mv <= 2.5f) {
        val y = baseY - mv * pxPerMv
        if (y in 0f..height) drawLine(major, Offset(0f, y), Offset(width, y), strokeWidth = 1f)
        mv += 0.5f
    }
    val i0 = w.t.binarySearch(rowStart).let { if (it < 0) -it - 1 else it }
    val i1 = w.t.binarySearch(rowEnd).let { if (it < 0) -it - 1 else it }
    val median = if (i1 > i0) w.mv.copyOfRange(i0, i1).sortedArray()[(i1 - i0) / 2] else 0f
    val xOf = { t: Long -> (t - rowStart) / 1e9f * pxPerS }
    val yOf = { t: Long -> EcgSnap.valueAt(w, t)?.let { baseY - (it - median) * pxPerMv } ?: baseY }
    if (i1 > i0) {
        val path = Path()
        var prev = Long.MIN_VALUE
        for (i in i0 until i1) {
            val x = xOf(w.t[i])
            val y = baseY - (w.mv[i] - median) * pxPerMv
            if (prev == Long.MIN_VALUE || w.t[i] - prev > 50_000_000L) path.moveTo(x, y) else path.lineTo(x, y)
            prev = w.t[i]
        }
        clipRect { drawPath(path, Color.Black, style = Stroke(width = 1.6f)) }
    }
    if (markerNs in rowStart until rowEnd) {
        val x = xOf(markerNs)
        drawLine(Color(0xFFD32F2F), Offset(x, 0f), Offset(x, height), strokeWidth = 3f)
    }
    val label = TextStyle(fontSize = 11.sp, color = MEASURE_BLUE, background = Color.White)
    for (m in measurements) {
        val a = minOf(m.aNs, m.bNs)
        val b = maxOf(m.aNs, m.bNs)
        if (b < rowStart || a >= rowEnd) continue
        val inA = a in rowStart until rowEnd
        val inB = b in rowStart until rowEnd
        val xa = if (inA) xOf(a) else 0f
        val xb = if (inB) xOf(b) else width
        val ys = listOfNotNull(if (inA) yOf(a) else null, if (inB) yOf(b) else null)
        val lineY = ((ys.minOrNull() ?: 40f) - 18f).coerceAtLeast(34f)
        val dash = PathEffect.dashPathEffect(floatArrayOf(6f, 6f))
        if (inA) drawLine(MEASURE_BLUE, Offset(xa, yOf(a)), Offset(xa, lineY), strokeWidth = 2f, pathEffect = dash)
        if (inB) drawLine(MEASURE_BLUE, Offset(xb, yOf(b)), Offset(xb, lineY), strokeWidth = 2f, pathEffect = dash)
        drawLine(MEASURE_BLUE, Offset(xa, lineY), Offset(xb, lineY), strokeWidth = 3f)
        if (inA) drawPath(Path().apply { moveTo(xa, lineY); lineTo(xa + 14f, lineY - 7f); lineTo(xa + 14f, lineY + 7f); close() }, MEASURE_BLUE)
        if (inB) drawPath(Path().apply { moveTo(xb, lineY); lineTo(xb - 14f, lineY - 7f); lineTo(xb - 14f, lineY + 7f); close() }, MEASURE_BLUE)
        if (inA || !inB) {
            val text = "%.0f ms · %.0f bpm".format(m.intervalMs, m.bpm) + if (!inB) " →" else ""
            val layout = measurer.measure(text, label)
            val tx = ((xa + xb) / 2 - layout.size.width / 2).coerceIn(2f, (width - layout.size.width - 2f).coerceAtLeast(2f))
            drawText(layout, topLeft = Offset(tx, lineY - layout.size.height - 4f))
        }
    }
    if (pending != null && pending in rowStart until rowEnd) {
        drawCircle(MEASURE_BLUE, radius = 9f, center = Offset(xOf(pending), yOf(pending)))
    }
}
