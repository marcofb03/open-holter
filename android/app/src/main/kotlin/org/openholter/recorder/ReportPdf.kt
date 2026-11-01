package org.openholter.recorder

import android.content.Context
import android.content.Intent
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import androidx.core.content.FileProvider
import org.openholter.core.EcgRows
import org.openholter.core.EcgSnap
import org.openholter.core.EcgWindow
import org.openholter.core.HrBin
import org.openholter.core.Marker
import org.openholter.core.Measurement
import org.openholter.core.RecordingFiles
import org.openholter.core.SignalReaders
import org.openholter.core.Training
import java.io.File
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.ceil
import kotlin.math.floor

/**
 * PDF reports for a doctor, A4 landscape:
 * - ECG: standard ECG paper scale (25 mm/s, 10 mm/mV, 1 mm / 5 mm grid), 10 s
 *   per row, 4 rows per page, as many pages as needed; symptom markers as red
 *   lines. The scale is stated in the header (the owner prefers rows without a
 *   drawn calibration pulse). The signal is drawn as delivered by the H10.
 * - Heart-rate summary of a training: one page, time on x, bpm on y.
 *
 * Text uses the bundled static Roboto 2.138 (assets/fonts, Apache 2.0): with
 * the phone's variable system font, Android's PDF writer embeds Type 3 fonts,
 * from which PDF viewers copy text with missing or extra spaces.
 */
object ReportPdf {
    private const val PT_PER_MM = 72f / 25.4f
    private const val PAGE_W = 842 // A4 landscape in points
    private const val PAGE_H = 595
    private const val MM_PER_S = 25f
    private const val MM_PER_MV = 10f
    private const val SECONDS_PER_ROW = 10
    private const val ROWS_PER_PAGE = 4
    private const val ROW_GAP_PT = 16f
    private const val TOP_PT = 112f
    private const val MARGIN = 40f

    /** Half an ECG row: the shifted marker PDF starts its rows this much earlier, so the marker is mid-row. */
    const val HALF_ROW_NS = SECONDS_PER_ROW * 1_000_000_000L / 2

    /** Longest ECG export (about 270 pages); longer trainings get the summary only. */
    const val MAX_ECG_NS = 3 * 3_600_000_000_000L

    private val DATE_TIME: DateTimeFormatter = DateTimeFormatter.ofPattern("EEE d MMM yyyy, HH:mm:ss")
    private val CLOCK: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss")
    private val FILE_STAMP: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")
    private const val SIGNAL =
        "Polar H10 chest strap, single lead, 130 Hz. Signal as delivered by the H10, no extra filtering by the app " +
            "(the H10's internal filtering is not documented by Polar). Scale 25 mm/s, 10 mm/mV. " +
            "\"bpm\" in a row: H10 heart rate, median of that row's readings with skin contact. " +
            "mV scale left of each row: 0 is the row's baseline (median of its signal)."
    private const val FOOTER = "Recorded with open-holter (open source). Not a medical device, not for diagnosis on its own."

    fun ecgPages(fromNs: Long, toNs: Long): Int =
        maxOf(1, ceil(ceil((toNs - fromNs) / (SECONDS_PER_ROW * 1e9)) / ROWS_PER_PAGE).toInt())

    /**
     * Rows: [upMm] above and [downMm] below the baseline (multiples of 5 mm, so the baseline
     * is on a thick grid line); always 10 mm/mV, so a taller signal gets taller rows and fewer
     * rows per page.
     */
    private class RowLayout(val upMm: Float, val downMm: Float) {
        val rowMm: Float get() = upMm + downMm
        val heightPt: Float get() = rowMm * PT_PER_MM
        val rowsPerPage: Int get() = maxOf(1, ((PAGE_H - 22f - TOP_PT) / (heightPt + ROW_GAP_PT)).toInt())
    }

    /** The standard 35-mm row (2.0 mV up, 1.5 mV down), 4 rows per page; used for trainings. */
    private val STANDARD_ROWS = RowLayout(20f, 15f)

    /**
     * Rows tall enough that the highest and the lowest point of the whole strip stay inside
     * the grid, plus room for the marker label and arrows on top and the bpm label below.
     * A strip taller than one page is clipped at one row per page.
     */
    private fun fittedRows(w: EcgWindow, fromNs: Long, toNs: Long, rowNs: Long, extraTopMm: Float): RowLayout {
        val (up, down) = EcgRows.extent(w, fromNs, toNs, rowNs)
        val round5 = { mm: Float -> (ceil(mm / 5.0) * 5).toFloat() }
        var upMm = round5(up * MM_PER_MV + extraTopMm).coerceAtLeast(15f)
        var downMm = round5(down * MM_PER_MV + 7f).coerceAtLeast(10f)
        val maxMm = (floor((PAGE_H - 22f - TOP_PT - ROW_GAP_PT) / PT_PER_MM / 5.0) * 5).toFloat()
        while (upMm + downMm > maxMm) if (upMm >= downMm) upMm -= 5f else downMm -= 5f
        return RowLayout(upMm, downMm)
    }

    /**
     * [shifted]: the caller widened the window by [HALF_ROW_NS] on both sides, so every row is
     * offset by 5 s and a beat at a row edge in the normal PDF is in the middle of a row here.
     */
    fun marker(
        context: Context, marker: Marker, w: EcgWindow, fromNs: Long, toNs: Long,
        shifted: Boolean = false, measurements: List<Measurement> = emptyList(),
    ): File {
        val zone = ZoneId.systemDefault()
        val beforeS = (marker.timestampNs - fromNs) / 1_000_000_000L
        val afterS = (toNs - marker.timestampNs) / 1_000_000_000L
        val info = buildList {
            add(
                "ECG from $beforeS s before to $afterS s after the marker" +
                    (if (shifted) " (rows shifted by 5 s, so the marker is in the middle of its row)" else "") +
                    ". $SIGNAL",
            )
            if (w.size == 0) add("No ECG was recorded around this marker (strap not connected or recording stopped).")
        }
        return ecg(
            context, "Symptom marker – ${DATE_TIME.format(local(marker.timestampNs, zone))} (${zone.id})",
            marker.comment, info, w, fromNs, toNs, listOf(marker.timestampNs),
            "open-holter_marker_${FILE_STAMP.format(local(marker.timestampNs, zone))}" +
                (if (shifted) "_shifted" else "") + (if (measurements.isNotEmpty()) "_measured" else ""),
            measurements,
            fitRows = true,
        )
    }

    fun trainingEcg(context: Context, t: Training, endNs: Long, w: EcgWindow, markers: List<Long>): File {
        val zone = ZoneId.systemDefault()
        val info = buildList {
            add("Whole training, ${duration(endNs - t.startNs)}. $SIGNAL" + if (markers.isNotEmpty()) " Red lines: symptom markers." else "")
            if (w.size == 0) add("No ECG was recorded during this training (strap not connected or recording stopped).")
        }
        return ecg(
            context, trainingTitle(t, endNs, zone), t.comment, info, w, t.startNs, endNs, markers,
            "open-holter_training_ecg_${FILE_STAMP.format(local(t.startNs, zone))}",
        )
    }

    /** One page: the H10 heart rate over the training ([bins] of [binNs]) and its numbers. */
    fun trainingSummary(context: Context, t: Training, endNs: Long, bins: List<HrBin>, binNs: Long, markers: List<Long>): File {
        val zone = ZoneId.systemDefault()
        val p = paints(context)
        val doc = PdfDocument()
        try {
            val page = doc.startPage(PdfDocument.PageInfo.Builder(PAGE_W, PAGE_H, 1).create())
            val c = page.canvas
            val info = buildList {
                add(
                    "Heart rate reported by the Polar H10 chest strap, median per ${binNs / 1_000_000_000L} s, " +
                        "only times with skin contact (the shaded band is the lowest and highest reading in each interval)." +
                        if (markers.isNotEmpty()) " Red lines: symptom markers." else "",
                )
                add(stats(bins, binNs, t.startNs, endNs))
            }
            header(c, p, trainingTitle(t, endNs, zone), t.comment, info)
            drawHrChart(c, p, bins, binNs, t.startNs, endNs, markers, zone, left = 70f, top = 125f, right = PAGE_W - MARGIN, bottom = PAGE_H - 60f)
            c.drawText("$FOOTER   Page 1 of 1", MARGIN, PAGE_H - 18f, p.small)
            doc.finishPage(page)
            return save(context, doc, "open-holter_training_summary_${FILE_STAMP.format(local(t.startNs, zone))}")
        } finally {
            doc.close()
        }
    }

    fun shareIntent(context: Context, file: File): Intent {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/pdf"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, file.nameWithoutExtension)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(send, "Share or save the report")
    }

    /** Lowest / median / highest of the bin medians, highest single reading, and coverage. */
    fun stats(bins: List<HrBin>, binNs: Long, fromNs: Long, toNs: Long): String {
        if (bins.isEmpty()) return "No heart rate with skin contact in this period."
        val med = bins.map { it.median }.sorted()
        val coveredMin = bins.size * binNs / 60e9
        return "Heart rate: lowest ${med.first()}, median ${med[med.size / 2]}, highest ${med.last()} bpm " +
            "(of the ${binNs / 1_000_000_000L}-s medians); highest single reading ${bins.maxOf { it.max }} bpm. " +
            "Data for %.0f of %.0f min.".format(coveredMin, (toNs - fromNs) / 60e9)
    }

    fun duration(ns: Long): String {
        val s = ns / 1_000_000_000L
        return if (s >= 3600) "%d h %02d min".format(s / 3600, s / 60 % 60) else "%d min %02d s".format(s / 60, s % 60)
    }

    // ---- ECG pages -------------------------------------------------------------------------

    private fun ecg(
        context: Context, title: String, comment: String, info: List<String>, w: EcgWindow,
        fromNs: Long, toNs: Long, markers: List<Long>, fileStem: String,
        measurements: List<Measurement> = emptyList(), fitRows: Boolean = false,
    ): File {
        val zone = ZoneId.systemDefault()
        val p = paints(context)
        val rowNs = SECONDS_PER_ROW * 1_000_000_000L
        val rows = ceil((toNs - fromNs).toDouble() / rowNs).toInt()
        val layout = if (fitRows) fittedRows(w, fromNs, toNs, rowNs, extraTopMm = if (measurements.isEmpty()) 5f else 9f) else STANDARD_ROWS
        val perPage = layout.rowsPerPage
        val pages = maxOf(1, ceil(rows.toDouble() / perPage).toInt())
        // H10 heart rate per row (bins aligned to the rows), for the "bpm" labels.
        val rowBpm = runCatching {
            SignalReaders.heartRate(Storage.root(context), zone, fromNs, fromNs + rows * rowNs, rowNs)
                .associate { it.startNs to it.median }
        }.getOrDefault(emptyMap())
        val doc = PdfDocument()
        try {
            for (pg in 0 until pages) {
                val page = doc.startPage(PdfDocument.PageInfo.Builder(PAGE_W, PAGE_H, pg + 1).create())
                val c = page.canvas
                header(c, p, title, comment, info, titleMaxW = SCALE_LEGEND_X - MARGIN - 10f)
                scaleLegend(c, p.small)
                for (k in 0 until perPage) {
                    val r = pg * perPage + k
                    if (r >= rows) break
                    val top = TOP_PT + k * (layout.heightPt + ROW_GAP_PT)
                    val rowStart = fromNs + r * rowNs
                    drawRow(c, w, rowStart, top, layout, markers, zone, p.small, measurements)
                    // bpm at the start of the strip (first row, left) and at the end of each
                    // row with a marker (right), inside the grid at the bottom, below the signal.
                    val bpm = "bpm: " + (rowBpm[rowStart]?.toString() ?: "–")
                    val bottom = top + layout.heightPt
                    if (r == 0) rowLabel(c, bpm, bottom, p.small, right = false)
                    if (markers.any { it in rowStart until rowStart + rowNs }) rowLabel(c, bpm, bottom, p.small, right = true)
                }
                c.drawText(
                    "$FOOTER Print at 100 % (no \"fit to page\") to keep 25 mm/s and 10 mm/mV.   Page ${pg + 1} of $pages",
                    MARGIN, PAGE_H - 18f, p.small,
                )
                doc.finishPage(page)
            }
            return save(context, doc, fileStem)
        } finally {
            doc.close()
        }
    }

    private fun drawRow(
        c: Canvas, w: EcgWindow, startNs: Long, top: Float, layout: RowLayout, markers: List<Long>, zone: ZoneId, label: Paint,
        measurements: List<Measurement> = emptyList(),
    ) {
        val left = (PAGE_W - SECONDS_PER_ROW * MM_PER_S * PT_PER_MM) / 2
        val width = SECONDS_PER_ROW * MM_PER_S * PT_PER_MM
        val height = layout.heightPt
        val endNs = startNs + SECONDS_PER_ROW * 1_000_000_000L
        c.drawText(CLOCK.format(local(startNs, zone)), left, top - 3f, label)

        val minor = Paint().apply { color = Color.rgb(246, 201, 201); strokeWidth = 0.25f }
        val major = Paint().apply { color = Color.rgb(229, 154, 154); strokeWidth = 0.6f }
        val mmX = (SECONDS_PER_ROW * MM_PER_S).toInt()
        for (i in 0..mmX) {
            val x = left + i * PT_PER_MM
            c.drawLine(x, top, x, top + height, if (i % 5 == 0) major else minor)
        }
        for (i in 0..layout.rowMm.toInt()) {
            val y = top + i * PT_PER_MM
            c.drawLine(left, y, left + width, y, if (i % 5 == 0) major else minor)
        }
        // mV scale in the margin, at every thick line: 0 is the row's baseline.
        val scale = Paint(label).apply { textSize = 6f; color = Color.GRAY; textAlign = Paint.Align.RIGHT }
        for (i in 0..layout.rowMm.toInt() step 5) {
            val mv = (layout.upMm - i) / MM_PER_MV
            val text = if (Math.abs(mv) < 1e-3) "0 mV" else String.format(java.util.Locale.ROOT, "%+.1f", mv)
            c.drawText(text, left - 3f, top + i * PT_PER_MM + 2f, scale)
        }

        val baseY = top + layout.upMm * PT_PER_MM
        val i0 = w.t.binarySearch(startNs).let { if (it < 0) -it - 1 else it }
        val i1 = w.t.binarySearch(endNs).let { if (it < 0) -it - 1 else it }
        val median = if (i1 > i0) w.mv.copyOfRange(i0, i1).sortedArray()[(i1 - i0) / 2] else 0f
        if (i1 > i0) {
            val path = Path()
            var prev = Long.MIN_VALUE
            for (i in i0 until i1) {
                val x = left + (w.t[i] - startNs) / 1e9f * MM_PER_S * PT_PER_MM
                val y = baseY - (w.mv[i] - median) * MM_PER_MV * PT_PER_MM
                if (prev == Long.MIN_VALUE || w.t[i] - prev > 50_000_000L) path.moveTo(x, y) else path.lineTo(x, y)
                prev = w.t[i]
            }
            val ink = Paint().apply {
                color = Color.BLACK; style = Paint.Style.STROKE; strokeWidth = 0.7f
                isAntiAlias = true; strokeJoin = Paint.Join.ROUND
            }
            c.save()
            c.clipRect(left, top, left + width, top + height)
            c.drawPath(path, ink)
            c.restore()
        } else {
            c.drawText("no ECG in this part", left + 6f, top + height / 2, label)
        }
        for (m in markers) {
            if (m !in startNs until endNs) continue
            val x = left + (m - startNs) / 1e9f * MM_PER_S * PT_PER_MM
            val red = Paint().apply {
                color = Color.rgb(211, 47, 47); strokeWidth = 1.5f; textSize = 8f; typeface = label.typeface; isAntiAlias = true
            }
            c.drawLine(x, top, x, top + height, red)
            c.drawText("MARKER ${CLOCK.format(local(m, zone))}", x + 3f, top + 9f, red)
        }
        val xOf = { t: Long -> left + (t - startNs) / 1e9f * MM_PER_S * PT_PER_MM }
        val yOf = { t: Long -> EcgSnap.valueAt(w, t)?.let { baseY - (it - median) * MM_PER_MV * PT_PER_MM } ?: baseY }
        for (m in measurements) {
            if (m.isAmplitude) drawAmplitude(c, w, m.aNs, startNs, endNs, top, height, median, baseY, xOf, label)
            else drawMeasurement(c, m, startNs, endNs, left, width, top, xOf, yOf, label)
        }
    }

    /**
     * The mV of one point (above or below the row's baseline) as a small black number next to
     * the wave, never on the trace: above a top, below a bottom (a freely placed point: above if
     * above the baseline), with a light dashed line from just above (or under) the point to it.
     */
    private fun drawAmplitude(
        c: Canvas, w: EcgWindow, tNs: Long, rowStart: Long, rowEnd: Long, top: Float, height: Float,
        median: Float, baseY: Float, xOf: (Long) -> Float, label: Paint,
    ) {
        if (tNs !in rowStart until rowEnd) return
        val v = EcgSnap.valueAt(w, tNs) ?: return
        val d = v - median
        val above = EcgSnap.kind(w, tNs) ?: (d >= 0)
        val yOf = { mv: Float -> baseY - (mv - median) * MM_PER_MV * PT_PER_MM }
        val y = yOf(v)
        // The trace's extreme around the point, so the number stays clear of it.
        val i0 = w.t.binarySearch(tNs - 60_000_000L).let { if (it < 0) -it - 1 else it }
        val i1 = w.t.binarySearch(tNs + 60_000_001L).let { if (it < 0) -it - 1 else it }
        var yExt = y
        for (i in i0 until i1) yExt = if (above) minOf(yExt, yOf(w.mv[i])) else maxOf(yExt, yOf(w.mv[i]))
        val x = xOf(tNs)
        val text = String.format(java.util.Locale.ROOT, "%+.2f mV", d)
        val tp = Paint(label).apply { color = Color.BLACK; textSize = 6.5f }
        val tw = tp.measureText(text)
        val minLead = 9f
        val textY = if (above) minOf(yExt - 4f, y - minLead).coerceAtLeast(top + 7f)
        else (maxOf(yExt + 4f, y + minLead) + 6f).coerceAtMost(top + height - 2f)
        val lead = Paint().apply {
            color = Color.argb(170, 0, 0, 0); strokeWidth = 0.5f; isAntiAlias = true
            pathEffect = android.graphics.DashPathEffect(floatArrayOf(1.5f, 1f), 0f)
        }
        val leadFrom = if (above) y - 1.5f else y + 1.5f
        val leadTo = if (above) textY + 1.5f else textY - 6.5f
        c.drawLine(x, leadFrom, x, leadTo, lead)
        c.drawText(text, (x - tw / 2).coerceAtLeast(xOf(rowStart) + 1f), textY, tp)
    }

    /**
     * A measurement as a horizontal double arrow just above the higher of its two points, with
     * dotted lines down to the points and "812 ms · 74 bpm". If it continues into the next row,
     * this row's part runs to the edge (label with "→") and the next row's part starts at the edge.
     */
    private fun drawMeasurement(
        c: Canvas, m: Measurement, rowStart: Long, rowEnd: Long, left: Float, width: Float, top: Float,
        xOf: (Long) -> Float, yOf: (Long) -> Float, label: Paint,
    ) {
        val a = minOf(m.aNs, m.bNs)
        val b = maxOf(m.aNs, m.bNs)
        if (b < rowStart || a >= rowEnd) return
        val inA = a in rowStart until rowEnd
        val inB = b in rowStart until rowEnd
        val xa = if (inA) xOf(a) else left
        val xb = if (inB) xOf(b) else left + width
        val ys = listOfNotNull(if (inA) yOf(a) else null, if (inB) yOf(b) else null)
        val lineY = ((ys.minOrNull() ?: (top + 20f)) - 7f).coerceAtLeast(top + 14f)
        val blue = Paint().apply {
            color = Color.rgb(21, 101, 192); strokeWidth = 0.9f; isAntiAlias = true; style = Paint.Style.STROKE
        }
        val dotted = Paint(blue).apply { strokeWidth = 0.6f; pathEffect = android.graphics.DashPathEffect(floatArrayOf(1.5f, 1.5f), 0f) }
        if (inA) c.drawLine(xa, yOf(a), xa, lineY, dotted)
        if (inB) c.drawLine(xb, yOf(b), xb, lineY, dotted)
        c.drawLine(xa, lineY, xb, lineY, blue)
        val head = Paint(blue).apply { style = Paint.Style.FILL }
        if (inA) c.drawPath(Path().apply { moveTo(xa, lineY); lineTo(xa + 4f, lineY - 2f); lineTo(xa + 4f, lineY + 2f); close() }, head)
        if (inB) c.drawPath(Path().apply { moveTo(xb, lineY); lineTo(xb - 4f, lineY - 2f); lineTo(xb - 4f, lineY + 2f); close() }, head)
        if (inA || !inB) {
            val text = "%.0f ms · %.0f bpm".format(m.intervalMs, m.bpm) + if (!inB) " →" else ""
            val tp = Paint(label).apply { color = Color.rgb(21, 101, 192); textSize = 7.5f }
            val tw = tp.measureText(text)
            val tx = ((xa + xb) / 2 - tw / 2).coerceIn(left + 2f, left + width - tw - 2f)
            c.drawRect(tx - 1f, lineY - 9.5f, tx + tw + 1f, lineY - 1.5f, Paint().apply { color = Color.WHITE })
            c.drawText(text, tx, lineY - 3f, tp)
        }
    }

    /** A small label on a white patch at the bottom left or right inside an ECG row. */
    private fun rowLabel(c: Canvas, text: String, bottom: Float, paint: Paint, right: Boolean) {
        val left = (PAGE_W - SECONDS_PER_ROW * MM_PER_S * PT_PER_MM) / 2
        val width = SECONDS_PER_ROW * MM_PER_S * PT_PER_MM
        val tw = paint.measureText(text)
        val x = if (right) left + width - tw - 4f else left + 4f
        val bg = Paint().apply { color = Color.WHITE }
        c.drawRect(x - 2f, bottom - 13f, x + tw + 2f, bottom - 2f, bg)
        val ink = Paint(paint).apply { color = Color.BLACK }
        c.drawText(text, x, bottom - 4.5f, ink)
    }

    // ---- heart-rate chart ------------------------------------------------------------------

    private fun drawHrChart(
        c: Canvas, p: Paints, bins: List<HrBin>, binNs: Long, fromNs: Long, toNs: Long, markers: List<Long>, zone: ZoneId,
        left: Float, top: Float, right: Float, bottom: Float,
    ) {
        val gridPaint = Paint().apply { color = Color.rgb(210, 210, 210); strokeWidth = 0.5f }
        val axis = Paint().apply { color = Color.DKGRAY; strokeWidth = 0.8f }
        val w = right - left
        val h = bottom - top
        val medians = bins.map { it.median }
        var lo = if (medians.isEmpty()) 40 else floor((medians.min() - 10) / 10.0).toInt() * 10
        var hi = if (medians.isEmpty()) 180 else ceil((medians.max() + 10) / 10.0).toInt() * 10
        if (hi - lo < 40) {
            val mid = (hi + lo) / 2
            lo = mid - 20
            hi = mid + 20
        }
        val step = if (hi - lo > 100) 20 else 10
        val xOf = { t: Long -> left + (t - fromNs).toFloat() / (toNs - fromNs) * w }
        val yOf = { bpm: Int -> bottom - (bpm - lo).toFloat() / (hi - lo) * h }

        var g = lo
        while (g <= hi) {
            val y = yOf(g)
            c.drawLine(left, y, right, y, gridPaint)
            c.drawText("$g", left - 22f, y + 3f, p.small)
            g += step
        }
        c.drawText("bpm", left - 22f, top - 8f, p.small)
        val spanMin = (toNs - fromNs) / 60e9
        val tickMin = listOf(1, 2, 5, 10, 15, 30, 60, 120, 180, 240).firstOrNull { spanMin / it <= 14 } ?: 360
        var tick = Instant.ofEpochSecond(fromNs / 1_000_000_000L).atZone(zone).withSecond(0).withNano(0)
        val off = (tick.hour * 60 + tick.minute) % tickMin
        if (off != 0) tick = tick.plusMinutes((tickMin - off).toLong())
        while (true) {
            val t = tick.toInstant().epochSecond * 1_000_000_000L
            if (t > toNs) break
            if (t >= fromNs) {
                val x = xOf(t)
                c.drawLine(x, top, x, bottom, gridPaint)
                c.drawText("%02d:%02d".format(tick.hour, tick.minute), x - 11f, bottom + 12f, p.small)
            }
            tick = tick.plusMinutes(tickMin.toLong())
        }
        c.drawLine(left, bottom, right, bottom, axis)
        c.drawLine(left, top, left, bottom, axis)

        val blue = Color.rgb(21, 101, 192)
        val binW = maxOf(0.6f, w * binNs.toFloat() / (toNs - fromNs))
        val band = Paint().apply { color = blue; alpha = 60; strokeWidth = binW }
        for (b in bins) {
            val x = xOf(b.startNs) + binW / 2
            c.drawLine(x, yOf(b.max.coerceAtMost(hi)), x, yOf(b.min.coerceAtLeast(lo)), band)
        }
        val line = Paint().apply {
            color = blue; style = Paint.Style.STROKE; strokeWidth = 1.4f; isAntiAlias = true; strokeJoin = Paint.Join.ROUND
        }
        val path = Path()
        var prev = Long.MIN_VALUE
        for (b in bins) {
            val x = xOf(b.startNs) + binW / 2
            val y = yOf(b.median.coerceIn(lo, hi))
            if (prev == Long.MIN_VALUE || b.startNs - prev > maxOf(2 * binNs, 30_000_000_000L)) path.moveTo(x, y) else path.lineTo(x, y)
            prev = b.startNs
        }
        c.drawPath(path, line)
        val red = Paint().apply { color = Color.rgb(211, 47, 47); strokeWidth = 1.2f; textSize = 8f; typeface = p.small.typeface; isAntiAlias = true }
        for (m in markers) {
            if (m !in fromNs..toNs) continue
            val x = xOf(m)
            c.drawLine(x, top, x, bottom, red)
            c.drawText("MARKER ${CLOCK.format(local(m, zone))}", x + 3f, top + 9f, red)
        }
        if (bins.isEmpty()) c.drawText("No heart rate with skin contact in this period.", left + 10f, top + h / 2, p.text)
    }

    // ---- common ----------------------------------------------------------------------------

    private class Paints(val title: Paint, val text: Paint, val small: Paint)

    private fun paints(context: Context): Paints {
        val regular = font(context, "fonts/Roboto-Regular.ttf", Typeface.NORMAL)
        val bold = font(context, "fonts/Roboto-Bold.ttf", Typeface.BOLD)
        return Paints(
            title = Paint().apply { textSize = 14f; typeface = bold; isAntiAlias = true },
            text = Paint().apply { textSize = 9.5f; typeface = regular; isAntiAlias = true },
            small = Paint().apply { textSize = 8f; typeface = regular; color = Color.DKGRAY; isAntiAlias = true },
        )
    }

    /** Title, the comment (at most 2 lines) and the description (always shown), at most 5 lines in all. */
    private fun header(c: Canvas, p: Paints, title: String, comment: String, info: List<String>, titleMaxW: Float = PAGE_W - 2 * MARGIN) {
        val maxW = PAGE_W - 2 * MARGIN
        val commentLines = wrap("Comment: " + comment.ifBlank { "(none)" }, p.text, maxW).let {
            if (it.size > 2) listOf(it[0], it[1].dropLast(1) + "…") else it
        }
        val lines = commentLines + info.flatMap { wrap(it, p.text, maxW) }
        // A long title gets a smaller font instead of running into the scale legend.
        val titlePaint = Paint(p.title)
        while (titlePaint.measureText(title) > titleMaxW && titlePaint.textSize > 9f) titlePaint.textSize -= 0.5f
        c.drawText(title, MARGIN, 40f, titlePaint)
        lines.take(5).forEachIndexed { i, line -> c.drawText(line, MARGIN, 56f + i * 11.5f, p.text) }
    }

    /** Left edge of the scale legend in the top right corner of ECG pages. */
    private const val SCALE_LEGEND_X = PAGE_W - MARGIN - 270f

    /**
     * One large square of the grid (5 x 5 small squares) in the top right corner, with what it
     * measures: 25 mm/s gives 40 ms per mm, 10 mm/mV gives 0.1 mV per mm.
     */
    private fun scaleLegend(c: Canvas, small: Paint) {
        val mm = PT_PER_MM
        val x0 = SCALE_LEGEND_X
        val y0 = 16f
        val minor = Paint().apply { color = Color.rgb(246, 201, 201); strokeWidth = 0.25f }
        val major = Paint().apply { color = Color.rgb(229, 154, 154); strokeWidth = 0.6f }
        for (i in 0..5) {
            val p = if (i == 0 || i == 5) major else minor
            c.drawLine(x0 + i * mm, y0, x0 + i * mm, y0 + 5 * mm, p)
            c.drawLine(x0, y0 + i * mm, x0 + 5 * mm, y0 + i * mm, p)
        }
        val ink = Paint(small).apply { color = Color.BLACK }
        val tx = x0 + 5 * mm + 6f
        c.drawText("1 large square (5 × 5 small) = 200 ms wide, 0.5 mV high", tx, y0 + 6f, ink)
        c.drawText("1 small square = 40 ms wide, 0.1 mV high", tx, y0 + 15f, ink)
        c.drawText("(25 mm/s, 10 mm/mV)", tx, y0 + 24f, ink)
    }

    private fun trainingTitle(t: Training, endNs: Long, zone: ZoneId): String =
        "Training – ${DATE_TIME.format(local(t.startNs, zone))} to ${CLOCK.format(local(endNs, zone))} " +
            "(${duration(endNs - t.startNs)}, ${zone.id})" + if (t.running) ", still running" else ""

    private fun save(context: Context, doc: PdfDocument, stem: String): File {
        val dir = File(context.cacheDir, "reports").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() } // only the latest export is kept
        val file = File(dir, "$stem.pdf")
        file.outputStream().use { doc.writeTo(it) }
        return file
    }

    private fun local(ns: Long, zone: ZoneId) = RecordingFiles.instantOf(ns).atZone(zone)

    private fun font(context: Context, asset: String, fallbackStyle: Int): Typeface =
        runCatching { Typeface.createFromAsset(context.assets, asset) }
            .getOrElse { Typeface.defaultFromStyle(fallbackStyle) }

    /** Splits [s] into lines no wider than [maxWidth] (by words). */
    private fun wrap(s: String, paint: Paint, maxWidth: Float): List<String> {
        val out = mutableListOf<String>()
        var line = StringBuilder()
        for (word in s.split(' ')) {
            val candidate = if (line.isEmpty()) word else "$line $word"
            if (paint.measureText(candidate) > maxWidth && line.isNotEmpty()) {
                out += line.toString()
                line = StringBuilder(word)
            } else {
                line = StringBuilder(candidate)
            }
        }
        if (line.isNotEmpty()) out += line.toString()
        return out
    }
}
