package org.openholter.recorder.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp

/**
 * Live ECG trace of the last ~10 s (130 Hz). Grid: 1 s vertical lines,
 * 0.5 mV horizontal lines. The vertical scale adapts to the signal.
 */
@Composable
fun EcgTrace(samples: FloatArray, modifier: Modifier = Modifier, capacity: Int = 1300) {
    val line = MaterialTheme.colorScheme.primary
    val grid = MaterialTheme.colorScheme.outlineVariant
    Canvas(modifier.fillMaxWidth().height(160.dp)) {
        val w = size.width
        val h = size.height
        for (s in 0..10) {
            val x = w * s / 10f
            drawLine(grid, Offset(x, 0f), Offset(x, h), strokeWidth = 1f)
        }
        if (samples.size < 2) return@Canvas
        val sorted = samples.sortedArray()
        val median = sorted[sorted.size / 2]
        val lo = sorted[(sorted.size * 0.005).toInt()]
        val hi = sorted[((sorted.size - 1) * 0.995).toInt()]
        // At least ±1 mV visible so noise does not look like a signal.
        val half = maxOf(1f, (hi - median), (median - lo)) * 1.1f
        val yOf = { v: Float -> h / 2 - (v - median) / half * (h / 2) }
        var mv = -10f
        while (mv <= 10f) {
            val y = yOf(median + mv)
            if (y in 0f..h) drawLine(grid.copy(alpha = 0.4f), Offset(0f, y), Offset(w, y), strokeWidth = 1f)
            mv += 0.5f
        }
        val dx = w / (capacity - 1)
        val start = capacity - samples.size // right-align: newest sample at the right edge
        val path = Path()
        samples.forEachIndexed { i, v ->
            val x = (start + i) * dx
            val y = yOf(v).coerceIn(0f, h)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, line, style = Stroke(width = 2f))
    }
}
