package org.openholter.recorder.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RangeSlider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.openholter.core.HrBin
import org.openholter.core.SignalReaders
import org.openholter.recorder.Storage
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt

/** Epoch ns of [hours] (0–24, in steps of 0.5) after local midnight of [day]. */
private fun nsAt(day: LocalDate, hours: Float, zone: ZoneId): Long {
    val minutes = (hours * 60).roundToInt()
    val zdt = if (minutes >= 24 * 60) day.plusDays(1).atStartOfDay(zone) else day.atTime(minutes / 60, minutes % 60).atZone(zone)
    val i = zdt.toInstant()
    return i.epochSecond * 1_000_000_000L + i.nano
}

private fun hhmm(hours: Float): String {
    val m = (hours * 60).roundToInt()
    return "%02d:%02d".format(m / 60, m % 60)
}

private val DAY_FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("EEE d MMM yyyy")

/**
 * The H10's heart rate over one day, in 1-minute medians (line) with the
 * minute's lowest and highest reading (band). Only readings with skin contact.
 * For curiosity; not part of the analysis.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HeartRateScreen() {
    val context = LocalContext.current
    val zone = ZoneId.systemDefault()
    var day by rememberSaveable { mutableStateOf(LocalDate.now()) }
    var start by rememberSaveable { mutableStateOf(7f) }
    var end by rememberSaveable { mutableStateOf(24f) }
    var bins by remember { mutableStateOf<List<HrBin>?>(null) }
    var refresh by remember { mutableIntStateOf(0) }
    var picking by remember { mutableStateOf(false) }

    LaunchedEffect(day, refresh) {
        bins = null
        bins = withContext(Dispatchers.IO) {
            SignalReaders.heartRate(Storage.root(context), zone, nsAt(day, 0f, zone), nsAt(day, 24f, zone))
        }
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        NotMedicalBanner()
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(onClick = { day = day.minusDays(1) }) { Text("◀") }
            TextButton(onClick = { picking = true }, modifier = Modifier.weight(1f)) {
                Text(DAY_FORMAT.format(day), style = MaterialTheme.typography.titleMedium)
            }
            OutlinedButton(onClick = { day = day.plusDays(1) }, enabled = day < LocalDate.now()) { Text("▶") }
        }

        InfoCard("Time range: ${hhmm(start)} – ${hhmm(end)}") {
            RangeSlider(
                value = start..end,
                onValueChange = { r ->
                    // Keep at least half an hour.
                    if (r.endInclusive - r.start >= 0.5f) {
                        start = r.start
                        end = r.endInclusive
                    }
                },
                valueRange = 0f..24f,
                steps = 47,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                TextButton(onClick = { start = 7f; end = 24f }) { Text("07–24") }
                TextButton(onClick = { start = 0f; end = 7f }) { Text("Night 00–07") }
                TextButton(onClick = { start = 0f; end = 24f }) { Text("Whole day") }
                if (day == LocalDate.now()) TextButton(onClick = { refresh++ }) { Text("Refresh") }
            }
        }

        val fromNs = nsAt(day, start, zone)
        val toNs = nsAt(day, end, zone)
        InfoCard("Heart rate (H10, 1-minute medians)") {
            val all = bins
            when {
                all == null -> Text("Loading…")
                else -> {
                    val shown = all.filter { it.startNs in fromNs until toNs }
                    if (shown.isEmpty()) {
                        Text("No heart rate with skin contact in this time range.")
                    } else {
                        HrChart(shown, fromNs, toNs, zone)
                        val medians = shown.map { it.median }.sorted()
                        Text(
                            "Lowest ${medians.first()} · median ${medians[medians.size / 2]} · highest ${medians.last()} bpm " +
                                "(of the 1-minute medians)",
                        )
                        Text(
                            "Data for %.1f h of %.1f h".format(shown.size / 60.0, (toNs - fromNs) / 3.6e12),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
        }
        Text(
            "Heart rate as reported by the H10 strap (not from the ECG analysis). Times without skin contact are " +
                "left out, because the strap sends wrong values when it is off. The shaded band is the lowest and " +
                "highest reading in each minute.",
            style = MaterialTheme.typography.bodySmall,
        )
    }

    if (picking) {
        val state = rememberDatePickerState(
            initialSelectedDateMillis = day.atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli(),
        )
        DatePickerDialog(
            onDismissRequest = { picking = false },
            confirmButton = {
                TextButton(onClick = {
                    state.selectedDateMillis?.let {
                        day = Instant.ofEpochMilli(it).atZone(ZoneOffset.UTC).toLocalDate().coerceAtMost(LocalDate.now())
                    }
                    picking = false
                }) { Text("OK") }
            },
            dismissButton = { TextButton(onClick = { picking = false }) { Text("Cancel") } },
        ) { DatePicker(state = state) }
    }
}

@Composable
private fun HrChart(bins: List<HrBin>, fromNs: Long, toNs: Long, zone: ZoneId) {
    val line = MaterialTheme.colorScheme.primary
    val grid = MaterialTheme.colorScheme.outlineVariant
    val labelColor = MaterialTheme.colorScheme.onSurfaceVariant
    val measurer = rememberTextMeasurer()
    val labelStyle = TextStyle(fontSize = 10.sp, color = labelColor)
    val medians = bins.map { it.median }
    var lo = floor((medians.min() - 10) / 10.0).toInt() * 10
    var hi = ceil((medians.max() + 10) / 10.0).toInt() * 10
    if (hi - lo < 40) {
        val mid = (hi + lo) / 2
        lo = mid - 20
        hi = mid + 20
    }
    val step = if (hi - lo > 100) 20 else 10
    val spanH = (toNs - fromNs) / 3.6e12
    val tickH = when {
        spanH <= 6 -> 1
        spanH <= 12 -> 2
        else -> 3
    }
    Canvas(Modifier.fillMaxWidth().height(260.dp)) {
        val left = 34.dp.toPx()
        val bottom = 18.dp.toPx()
        val w = size.width - left
        val h = size.height - bottom
        val xOf = { t: Long -> left + (t - fromNs).toFloat() / (toNs - fromNs) * w }
        val yOf = { bpm: Int -> h - (bpm - lo).toFloat() / (hi - lo) * h }

        var g = lo
        while (g <= hi) {
            val y = yOf(g)
            drawLine(grid, Offset(left, y), Offset(size.width, y), strokeWidth = 1f)
            drawText(measurer, "$g", Offset(0f, y - 7.sp.toPx()), labelStyle)
            g += step
        }
        // Full local hours, every tickH hours.
        var tick = Instant.ofEpochSecond(fromNs / 1_000_000_000L).atZone(zone).withMinute(0).withSecond(0).withNano(0)
        while (true) {
            val i = tick.toInstant()
            val t = i.epochSecond * 1_000_000_000L
            if (t > toNs) break
            if (t >= fromNs && tick.hour % tickH == 0) {
                val x = xOf(t)
                drawLine(grid, Offset(x, 0f), Offset(x, h), strokeWidth = 1f)
                drawText(measurer, "%02d".format(tick.hour), Offset(x - 6.sp.toPx(), h + 2.dp.toPx()), labelStyle)
            }
            tick = tick.plusHours(1)
        }

        val binW = maxOf(1f, w * 60e9.toFloat() / (toNs - fromNs))
        val band = line.copy(alpha = 0.22f)
        for (b in bins) {
            val x = xOf(b.startNs) + binW / 2
            drawLine(
                band, Offset(x, yOf(b.max.coerceAtMost(hi))), Offset(x, yOf(b.min.coerceAtLeast(lo))),
                strokeWidth = binW,
            )
        }
        val path = Path()
        var prev = Long.MIN_VALUE
        for (b in bins) {
            val x = xOf(b.startNs) + binW / 2
            val y = yOf(b.median.coerceIn(lo, hi))
            // A gap of more than 2 minutes (no data / no contact) breaks the line.
            if (prev == Long.MIN_VALUE || b.startNs - prev > 120_000_000_000L) path.moveTo(x, y) else path.lineTo(x, y)
            prev = b.startNs
        }
        drawPath(path, line, style = Stroke(width = 2.dp.toPx()))
    }
}
