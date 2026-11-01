package org.openholter.recorder.ui

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.unit.sp
import org.openholter.core.EcgRows
import org.openholter.core.EcgSnap
import org.openholter.core.EcgWindow
import java.util.Locale
import kotlin.math.ceil

private val GRID_MINOR = Color(0xFFF6C9C9)
private val GRID_MAJOR = Color(0xFFE59A9A)

/** Phone rows: how many mV fit above and below the baseline (0.5-mV steps). */
class PaperRows(val upMv: Float, val downMv: Float) {
    val spanMv: Float get() = upMv + downMv
}

/**
 * Rows tall enough for the highest and lowest point of the strip (each row's signal is
 * drawn around its own baseline, the row median), with [extraTopMv] room above for labels.
 */
fun paperRows(w: EcgWindow, fromNs: Long, toNs: Long, rowNs: Long, extraTopMv: Float = 0.4f): PaperRows {
    val (up, down) = EcgRows.extent(w, fromNs, toNs, rowNs)
    val half = { mv: Float -> (ceil(mv / 0.5) * 0.5).toFloat() }
    var upMv = half(up + extraTopMv).coerceAtLeast(1.5f)
    var downMv = half(down + 0.3f).coerceAtLeast(1.0f)
    while (upMv + downMv > 8f) if (upMv >= downMv) upMv -= 0.5f else downMv -= 0.5f
    return PaperRows(upMv, downMv)
}

/** Where things are in a drawn row; x and y in pixels of the row's canvas. */
class PaperRow(
    val rowStart: Long,
    val rowNs: Long,
    val gutter: Float,
    val pxPerS: Float,
    val pxPerMv: Float,
    val baseY: Float,
    val median: Float,
) {
    fun xOf(t: Long): Float = gutter + (t - rowStart) / 1e9f * pxPerS
    fun yOfMv(mv: Float): Float = baseY - (mv - median) * pxPerMv
    fun timeAt(x: Float): Long = rowStart + ((x - gutter) / pxPerS * 1e9f).toLong()
}

/**
 * One row of ECG paper: the mV scale in a gutter on the left (0 = the row's baseline), the
 * grid (0.2 s and 0.5 mV thick lines, 0.04 s fine lines), and the signal in black.
 */
fun DrawScope.drawPaperRow(
    w: EcgWindow, rowStart: Long, rowNs: Long, rows: PaperRows, gutter: Float, measurer: TextMeasurer,
): PaperRow {
    val width = size.width - gutter
    val height = size.height
    val seconds = rowNs / 1e9f
    val pxPerS = width / seconds
    val pxPerMv = height / rows.spanMv
    val baseY = rows.upMv * pxPerMv
    val median = EcgRows.median(w, rowStart, rowStart + rowNs) ?: 0f
    val row = PaperRow(rowStart, rowNs, gutter, pxPerS, pxPerMv, baseY, median)

    drawRect(Color.White, topLeft = Offset(gutter, 0f), size = androidx.compose.ui.geometry.Size(width, height))
    // Fine vertical lines every 0.04 s only when they are at least 3 px apart.
    val fine = pxPerS * 0.04f >= 3f
    val steps = (seconds / 0.04f).toInt()
    for (k in 0..steps) {
        val major = k % 5 == 0
        if (!major && !fine) continue
        val x = gutter + k * 0.04f * pxPerS
        drawLine(if (major) GRID_MAJOR else GRID_MINOR, Offset(x, 0f), Offset(x, height), strokeWidth = if (k % 25 == 0) 2f else 1f)
    }
    val label = TextStyle(fontSize = 9.sp, color = Color.Gray)
    var mv = -rows.downMv
    while (mv <= rows.upMv + 1e-3f) {
        val y = baseY - mv * pxPerMv
        drawLine(GRID_MAJOR, Offset(gutter, y), Offset(size.width, y), strokeWidth = 1f)
        val text = if (Math.abs(mv) < 1e-3f) "0 mV" else String.format(Locale.ROOT, "%+.1f", mv)
        val layout = measurer.measure(text, label)
        drawText(layout, topLeft = Offset(gutter - layout.size.width - 3f, (y - layout.size.height / 2f).coerceIn(0f, height - layout.size.height)))
        mv += 0.5f
    }

    val i0 = w.t.binarySearch(rowStart).let { if (it < 0) -it - 1 else it }
    val i1 = w.t.binarySearch(rowStart + rowNs).let { if (it < 0) -it - 1 else it }
    if (i1 > i0) {
        val path = Path()
        var prev = Long.MIN_VALUE
        for (i in i0 until i1) {
            val x = row.xOf(w.t[i])
            val y = row.yOfMv(w.mv[i])
            if (prev == Long.MIN_VALUE || w.t[i] - prev > 50_000_000L) path.moveTo(x, y) else path.lineTo(x, y)
            prev = w.t[i]
        }
        clipRect(left = gutter) { drawPath(path, Color.Black, style = Stroke(width = 1.6f)) }
    }
    return row
}

/**
 * The mV of one point above or below the row's baseline: a small black number next to the
 * wave, above a top and below a bottom (for a point placed freely: above if it is above the
 * baseline), clear of the trace, with a light dashed line from just above (or under) the
 * point to the number, so it is clear which point it belongs to.
 */
fun DrawScope.drawAmplitudeLabel(w: EcgWindow, row: PaperRow, tNs: Long, measurer: TextMeasurer) {
    if (tNs !in row.rowStart until row.rowStart + row.rowNs) return
    val v = EcgSnap.valueAt(w, tNs) ?: return
    val d = v - row.median
    val above = EcgSnap.kind(w, tNs) ?: (d >= 0)
    val y = row.yOfMv(v)
    // Keep the number clear of the trace around the point.
    val i0 = w.t.binarySearch(tNs - 60_000_000L).let { if (it < 0) -it - 1 else it }
    val i1 = w.t.binarySearch(tNs + 60_000_001L).let { if (it < 0) -it - 1 else it }
    var yExt = y
    for (i in i0 until i1) {
        val yi = row.yOfMv(w.mv[i])
        yExt = if (above) minOf(yExt, yi) else maxOf(yExt, yi)
    }
    val layout = measurer.measure(String.format(Locale.ROOT, "%+.2f mV", d), TextStyle(fontSize = 10.sp, color = Color.Black))
    val x = row.xOf(tNs)
    val minLead = 28f
    val textTop = if (above) (minOf(yExt - 8f, y - minLead) - layout.size.height).coerceAtLeast(0f)
    else maxOf(yExt + 8f, y + minLead).coerceAtMost(size.height - layout.size.height)
    val leadFrom = if (above) y - 4f else y + 4f
    val leadTo = if (above) textTop + layout.size.height + 1f else textTop - 1f
    drawLine(
        Color.Black.copy(alpha = 0.6f), Offset(x, leadFrom), Offset(x, leadTo),
        strokeWidth = 1.5f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(6f, 4f)),
    )
    val tx = (x - layout.size.width / 2f).coerceIn(row.gutter, size.width - layout.size.width)
    drawText(layout, topLeft = Offset(tx, textTop))
}
