package org.openholter.recorder.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
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
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.openholter.core.EcgRows
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
private val GUTTER = 34.dp

/**
 * Measure intervals on the ECG of a marker: the same 10-s rows as the PDF (normal or
 * shifted by half a row), landscape (10-s rows) or portrait (7-s rows), each row as tall as its signal. Tap two points; each one snaps to the tip of the beat within 50 ms
 * (or stays where tapped in "free" mode) and a blue arrow shows the time between them in
 * ms and as a rate in bpm. Measurements are saved with the marker and go into
 * "Export PDF with my measurements".
 */
@Composable
fun PersonalizeScreen(marker: Marker, fromNs: Long, toNs: Long, onBack: () -> Unit) {
    val context = LocalContext.current
    val app = HolterApp.from(context)
    val zone = ZoneId.systemDefault()
    val scope = rememberCoroutineScope()
    var window by remember { mutableStateOf<EcgWindow?>(null) }
    var measurements by remember { mutableStateOf<List<Measurement>>(emptyList()) }
    var snapToPeak by remember { mutableStateOf(true) }
    var amplitudeMode by remember { mutableStateOf(false) } // tap = mV label instead of a time measurement
    var shifted by remember { mutableStateOf(false) }
    var sheet by remember { mutableStateOf(ReportPdf.Sheet.LANDSCAPE) }
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

    val rowNs = sheet.rowNs
    val rowsFrom = if (shifted) fromNs - sheet.halfRowNs else fromNs
    val rowsTo = if (shifted) toNs + sheet.halfRowNs else toNs
    val rows = ceil((rowsTo - rowsFrom).toDouble() / rowNs).toInt()
    val rowStartOf = { t: Long -> rowsFrom + Math.floorDiv(t - rowsFrom, rowNs) * rowNs }

    /**
     * A tap at time [tNs] and height [tapMv]. With "Snap to peak" the point goes to the top or
     * bottom of the wave nearest to the finger on the screen ([nsPerPx], [mvPerPx]), within 150 ms.
     */
    fun tap(tNs: Long, tapMv: Float, nsPerPx: Double, mvPerPx: Double) {
        val w = window ?: return
        val snapped = { if (snapToPeak) EcgSnap.nearestExtremum(w, tNs, tapMv, nsPerPx, mvPerPx)?.tNs else EcgSnap.nearest(w, tNs) }
        if (amplitudeMode) {
            pending = null
            val point = snapped() ?: return
            val m = Measurement(point, point)
            if (m !in measurements) {
                measurements = (measurements + m).sortedBy { it.aNs }
                scope.launch { withContext(Dispatchers.IO) { app.measurements.add(ts, m) } }
            }
            return
        }
        val point = snapped() ?: return
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
            when {
                amplitudeMode -> "Tap a wave to label its height in mV."
                pending == null -> "Tap a point on the ECG, then a second one."
                else -> "Now tap the second point."
            },
            style = MaterialTheme.typography.bodyMedium,
            color = if (pending == null) MaterialTheme.colorScheme.onSurface else MEASURE_BLUE,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            FilterChip(selected = !amplitudeMode, onClick = { amplitudeMode = false }, label = { Text("Time (ms)") })
            FilterChip(selected = amplitudeMode, onClick = { amplitudeMode = true; pending = null }, label = { Text("Amplitude (mV)") })
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            FilterChip(selected = snapToPeak, onClick = { snapToPeak = true }, label = { Text("Snap to peak") })
            FilterChip(selected = !snapToPeak, onClick = { snapToPeak = false }, label = { Text("Free") })
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            FilterChip(selected = !shifted, onClick = { shifted = false }, label = { Text("Rows as PDF") })
            FilterChip(selected = shifted, onClick = { shifted = true }, label = { Text("Shifted ½ row") })
            listOf(1, 2, 4).forEach { z -> FilterChip(selected = zoom == z, onClick = { zoom = z }, label = { Text("${z}×") }) }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            FilterChip(
                selected = sheet == ReportPdf.Sheet.LANDSCAPE, onClick = { sheet = ReportPdf.Sheet.LANDSCAPE; pending = null },
                label = { Text("Landscape · 10-s rows") },
            )
            FilterChip(
                selected = sheet == ReportPdf.Sheet.PORTRAIT, onClick = { sheet = ReportPdf.Sheet.PORTRAIT; pending = null },
                label = { Text("Portrait · 7-s rows") },
            )
        }

        val w = window
        if (w == null) {
            Text("Loading…")
        } else if (w.size == 0) {
            Text("No ECG was recorded around this marker.")
        } else {
            val measurer = rememberTextMeasurer()
            val gutterPx = with(LocalDensity.current) { GUTTER.toPx() }
            BoxWithConstraints(Modifier.fillMaxWidth()) {
                val rowWidth = maxWidth * zoom
                Box(Modifier.horizontalScroll(rememberScrollState())) {
                    Column(Modifier.width(rowWidth)) {
                        for (r in 0 until rows) {
                            val rowStart = rowsFrom + r * rowNs
                            // Each row as tall as its own signal; the same mV per dp in every row.
                            val paper = remember(w, rowStart, rowNs) { paperRows(w, rowStart, rowStart + rowNs, rowNs, extraTopMv = 0.9f, extraBottomMv = 0.9f) }
                            Text(
                                CLOCK_S.format(RecordingFiles.instantOf(rowStart).atZone(zone)),
                                style = MaterialTheme.typography.labelSmall,
                                modifier = Modifier.padding(top = 6.dp),
                            )
                            Canvas(
                                Modifier.fillMaxWidth().height((paper.spanMv * 40).dp)
                                    .pointerInput(rowStart, rowNs, snapToPeak, amplitudeMode, pending, window, shifted, paper) {
                                        detectTapGestures { pos ->
                                            if (pos.x >= gutterPx) {
                                                // Same geometry as drawPaperRow: time from x, mV from y.
                                                val rowW = size.width - gutterPx
                                                val pxPerMv = size.height / paper.spanMv
                                                val median = EcgRows.median(w, rowStart, rowStart + rowNs) ?: 0f
                                                val tapMv = median + (paper.upMv * pxPerMv - pos.y) / pxPerMv
                                                tap(
                                                    rowStart + ((pos.x - gutterPx) / rowW * rowNs).toLong(), tapMv,
                                                    nsPerPx = rowNs / rowW.toDouble(), mvPerPx = 1.0 / pxPerMv,
                                                )
                                            }
                                        }
                                    },
                            ) {
                                drawEcgRow(w, rowStart, rowNs, paper, gutterPx, ts, measurements, pending, measurer)
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
                            if (m.isAmplitude) {
                                val win = window
                                val base = win?.let { EcgRows.median(it, rowStartOf(m.aNs), rowStartOf(m.aNs) + rowNs) }
                                val v = win?.let { EcgSnap.valueAt(it, m.aNs) }
                                Text(
                                    "${i + 1}.  " + if (v != null && base != null) String.format(java.util.Locale.ROOT, "%+.2f mV", v - base) else "– mV",
                                )
                                Text(MS_CLOCK.format(RecordingFiles.instantOf(m.aNs).atZone(zone)), style = MaterialTheme.typography.labelSmall)
                            } else {
                                Text("${i + 1}.  %.0f ms · %.0f bpm".format(m.intervalMs, m.bpm), color = MEASURE_BLUE)
                                Text(
                                    MS_CLOCK.format(RecordingFiles.instantOf(m.aNs).atZone(zone)) + " → " +
                                        MS_CLOCK.format(RecordingFiles.instantOf(m.bNs).atZone(zone)),
                                    style = MaterialTheme.typography.labelSmall,
                                )
                            }
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
                        runCatching { ReportPdf.marker(context, marker, win, rowsFrom, rowsTo, shifted, measurements, sheet) }
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
"\"Amplitude\" labels a wave with its mV above (+) or below (−) the row's baseline (the median of the " +
                "row, 0 on the scale). \"Snap to peak\" puts each point on the top or the bottom of the wave nearest to " +
                "your finger (within 150 ms), e.g. the tip of a beat or the bottom of the dip after it. Tops are " +
                "labelled above, bottoms below. " +
                "Snapping is good for the time from beat to beat. \"Free\" keeps the point where you tap, e.g. for the " +
                "start and end of one beat; zoom in for that. The H10 records every 7.7 ms, so a measurement is " +
                "accurate to about ±8 ms. bpm = 60 000 ÷ ms. Not a medical device.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

private val CLOCK_S: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss")

/** One row: ECG paper with the mV scale, the signal, the marker, time measurements and mV labels. */
private fun DrawScope.drawEcgRow(
    w: EcgWindow, rowStart: Long, rowNs: Long, paper: PaperRows, gutter: Float, markerNs: Long, measurements: List<Measurement>,
    pending: Long?, measurer: androidx.compose.ui.text.TextMeasurer,
) {
    val row = drawPaperRow(w, rowStart, rowNs, paper, gutter, measurer)
    val rowEnd = rowStart + rowNs
    val width = size.width
    if (markerNs in rowStart until rowEnd) {
        val x = row.xOf(markerNs)
        drawLine(Color(0xFFD32F2F), Offset(x, 0f), Offset(x, size.height), strokeWidth = 3f)
    }
    val yAt = { t: Long -> EcgSnap.valueAt(w, t)?.let { row.yOfMv(it) } ?: row.baseY }
    val label = TextStyle(fontSize = 11.sp, color = MEASURE_BLUE, background = Color.White)
    for (m in measurements) {
        if (m.isAmplitude) {
            drawAmplitudeLabel(w, row, m.aNs, measurer)
            continue
        }
        val a = minOf(m.aNs, m.bNs)
        val b = maxOf(m.aNs, m.bNs)
        if (b < rowStart || a >= rowEnd) continue
        val inA = a in rowStart until rowEnd
        val inB = b in rowStart until rowEnd
        val xa = if (inA) row.xOf(a) else gutter
        val xb = if (inB) row.xOf(b) else width
        val ys = listOfNotNull(if (inA) yAt(a) else null, if (inB) yAt(b) else null)
        val lineY = ((ys.minOrNull() ?: 40f) - 18f).coerceAtLeast(34f)
        val dash = PathEffect.dashPathEffect(floatArrayOf(6f, 6f))
        if (inA) drawLine(MEASURE_BLUE, Offset(xa, yAt(a)), Offset(xa, lineY), strokeWidth = 2f, pathEffect = dash)
        if (inB) drawLine(MEASURE_BLUE, Offset(xb, yAt(b)), Offset(xb, lineY), strokeWidth = 2f, pathEffect = dash)
        drawLine(MEASURE_BLUE, Offset(xa, lineY), Offset(xb, lineY), strokeWidth = 3f)
        if (inA) drawPath(Path().apply { moveTo(xa, lineY); lineTo(xa + 14f, lineY - 7f); lineTo(xa + 14f, lineY + 7f); close() }, MEASURE_BLUE)
        if (inB) drawPath(Path().apply { moveTo(xb, lineY); lineTo(xb - 14f, lineY - 7f); lineTo(xb - 14f, lineY + 7f); close() }, MEASURE_BLUE)
        if (inA || !inB) {
            val text = "%.0f ms · %.0f bpm".format(m.intervalMs, m.bpm) + if (!inB) " →" else ""
            val layout = measurer.measure(text, label)
            val tx = ((xa + xb) / 2 - layout.size.width / 2).coerceIn(gutter, (width - layout.size.width - 2f).coerceAtLeast(gutter))
            drawText(layout, topLeft = Offset(tx, lineY - layout.size.height - 4f))
        }
    }
    if (pending != null && pending in rowStart until rowEnd) {
        drawCircle(MEASURE_BLUE, radius = 9f, center = Offset(row.xOf(pending), yAt(pending)))
    }
}
