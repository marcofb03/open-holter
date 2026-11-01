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
 * - ECG: standard ECG paper scale (25 mm/s, 10 mm/mV, 1 mm / 5 mm grid), A4
 *   landscape with 10-s rows or portrait with 7-s rows ([Sheet]); each row is as
 *   tall as its own signal needs, so rows differ in height but never in scale;
 *   as many pages as needed; symptom markers as red lines. The scale is stated in the header (the owner prefers rows without a
 *   drawn calibration pulse). The signal is drawn as delivered by the H10.
 * - Heart-rate summary of a training: one page, time on x, bpm on y.
 *
 * Text uses the bundled static Roboto 2.138 (assets/fonts, Apache 2.0): with
 * the phone's variable system font, Android's PDF writer embeds Type 3 fonts,
 * from which PDF viewers copy text with missing or extra spaces.
 */
object ReportPdf {
    private const val PT_PER_MM = 72f / 25.4f
    private const val PAGE_W = 842 // A4 landscape in points (heart-rate summary)
    private const val PAGE_H = 595
    private const val MM_PER_S = 25f
    private const val MM_PER_MV = 10f
    private const val ROWS_PER_PAGE = 4 // only for the page estimate of a training ECG
    private const val ROW_GAP_PT = 4f
    private const val LABEL_LINE_PT = 11f // the line above each row: time, bpm, marker
    private const val LABEL_PAD_PT = 12f // room outside the grid for arrows and mV numbers
    private const val MARGIN = 40f
    private const val FOOTER_ROOM_PT = 16f

    /**
     * Page format of the ECG PDFs, both at 25 mm/s: A4 landscape fits 10-s rows (250 mm),
     * A4 portrait 7-s rows (175 mm).
     */
    enum class Sheet(val pageW: Int, val pageH: Int, val rowSeconds: Int) {
        LANDSCAPE(842, 595, 10),
        PORTRAIT(595, 842, 7);

        val rowNs: Long get() = rowSeconds * 1_000_000_000L

        /** The shifted PDF starts its rows this much earlier, so the marker is mid-row. */
        val halfRowNs: Long get() = rowNs / 2
        val rowWidthPt: Float get() = rowSeconds * MM_PER_S * PT_PER_MM
        val leftPt: Float get() = (pageW - rowWidthPt) / 2
    }

    /** Half a landscape row (5 s), for the shifted marker PDF. */
    const val HALF_ROW_NS = 5_000_000_000L

    /** Longest ECG export (about 270 pages); longer trainings get the summary only. */
    const val MAX_ECG_NS = 3 * 3_600_000_000_000L

    private val DATE_TIME: DateTimeFormatter = DateTimeFormatter.ofPattern("EEE d MMM yyyy, HH:mm:ss")
    private val CLOCK: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss")
    private val FILE_STAMP: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")
    private const val SIGNAL =
        "Polar H10 chest strap, single lead, 130 Hz. Signal as delivered by the H10, no extra filtering by the app " +
            "(the H10's internal filtering is not documented by Polar). Scale 25 mm/s, 10 mm/mV. " +
            "\"bpm\" in a row: H10 heart rate, median of that row's readings with skin contact. " +
            "mV scale left of each row: raw values from the H10 (0 mV is the H10's own zero, the same in every row)."

    /** About how many landscape pages an ECG of this length takes (rows vary in height). */
    fun ecgPages(fromNs: Long, toNs: Long): Int =
        maxOf(1, ceil(ceil((toNs - fromNs) / 10e9) / ROWS_PER_PAGE).toInt())

    /**
     * One row: a grid of [upMm] above and [downMm] below 0 mV (multiples of 5 mm =
     * 0.5 mV, so 0 mV is on a thick line), at 10 mm/mV; [padTopPt] / [padBottomPt] of
     * room outside the grid for measurement arrows and mV numbers (0 on rows without them).
     */
    private class RowLayout(val upMm: Float, val downMm: Float, val padTopPt: Float, val padBottomPt: Float) {
        val rowMm: Float get() = upMm + downMm
        val heightPt: Float get() = rowMm * PT_PER_MM

        /** Everything the row takes on the page: label line, padding, grid. */
        val blockPt: Float get() = LABEL_LINE_PT + padTopPt + heightPt + padBottomPt
    }

    /**
     * The grid of a row from its own highest and lowest point, each rounded outwards to the
     * next 0.5 mV and no more (e.g. +2.37 / -1.67 mV gives +2.5 / -2.0 mV); at most [maxMm].
     */
    private fun rowLayout(w: EcgWindow, rowStart: Long, rowNs: Long, padTopPt: Float, padBottomPt: Float, maxMm: Float): RowLayout {
        val (up, down) = EcgRows.extent(w, rowStart, rowStart + rowNs, rowNs)
        val round5 = { mm: Float -> (ceil(mm / 5.0 - 1e-6) * 5).toFloat() }
        var upMm = round5(up * MM_PER_MV).coerceAtLeast(5f)
        var downMm = round5(down * MM_PER_MV).coerceAtLeast(5f)
        while (upMm + downMm > maxMm) if (upMm >= downMm) upMm -= 5f else downMm -= 5f
        return RowLayout(upMm, downMm, padTopPt, padBottomPt)
    }

    /**
     * [shifted]: the caller widened the window by half a row of [sheet] on both sides, so every
     * row is offset by half a row and a beat at a row edge in the normal PDF is mid-row here.
     */
    fun marker(
        context: Context, marker: Marker, w: EcgWindow, fromNs: Long, toNs: Long,
        shifted: Boolean = false, measurements: List<Measurement> = emptyList(), sheet: Sheet = Sheet.LANDSCAPE,
    ): File {
        val zone = ZoneId.systemDefault()
        val beforeS = (marker.timestampNs - fromNs) / 1_000_000_000L
        val afterS = (toNs - marker.timestampNs) / 1_000_000_000L
        val info = buildList {
            add(
                "ECG from $beforeS s before to $afterS s after the marker" +
                    (if (shifted) " (rows shifted by ${sheet.rowSeconds / 2.0} s, so the marker is in the middle of its row)" else "") +
                    ". $SIGNAL",
            )
            if (w.size == 0) add("No ECG was recorded around this marker (strap not connected or recording stopped).")
        }
        return ecg(
            context, "Symptom marker – ${DATE_TIME.format(local(marker.timestampNs, zone))} (${zone.id})",
            marker.comment, info, w, fromNs, toNs, listOf(marker.timestampNs),
            "open-holter_marker_${FILE_STAMP.format(local(marker.timestampNs, zone))}" +
                (if (shifted) "_shifted" else "") + (if (measurements.isNotEmpty()) "_measured" else "") +
                (if (sheet == Sheet.PORTRAIT) "_portrait" else ""),
            measurements,
            sheet,
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
            header(c, p, Sheet.LANDSCAPE, trainingTitle(t, endNs, zone), t.comment, info)
            drawHrChart(c, p, bins, binNs, t.startNs, endNs, markers, zone, left = 70f, top = 125f, right = PAGE_W - MARGIN, bottom = PAGE_H - 60f)
            c.drawText("Page 1 of 1", PAGE_W - MARGIN, PAGE_H - 8f, Paint(p.small).apply { textAlign = Paint.Align.RIGHT })
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
        measurements: List<Measurement> = emptyList(), sheet: Sheet = Sheet.LANDSCAPE,
    ): File {
        val zone = ZoneId.systemDefault()
        val p = paints(context)
        val rowNs = sheet.rowNs
        val rows = ceil((toNs - fromNs).toDouble() / rowNs).toInt()
        // H10 heart rate per row (bins aligned to the rows), for the "bpm" labels.
        val rowBpm = runCatching {
            SignalReaders.heartRate(Storage.root(context), zone, fromNs, fromNs + rows * rowNs, rowNs)
                .associate { it.startNs to it.median }
        }.getOrDefault(emptyMap())
        val rowsTop = headerBottom(p, sheet, comment, info) + 8f
        val maxMm = (floor((sheet.pageH - FOOTER_ROOM_PT - rowsTop - LABEL_LINE_PT - 2 * LABEL_PAD_PT) / PT_PER_MM / 5.0) * 5).toFloat()

        // Each row's own grid; room for labels only on rows that have measurements.
        val layouts = (0 until rows).map { r ->
            val a = fromNs + r * rowNs
            val b = a + rowNs
            val inRow = measurements.filter { minOf(it.aNs, it.bNs) < b && maxOf(it.aNs, it.bNs) >= a }
            val median = 0f // raw values: 0 mV is the H10's zero
            // mV numbers go above a top and below a bottom (as drawAmplitude does).
            val sides = inRow.filter { it.isAmplitude && it.aNs in a until b }
                .map { m -> EcgSnap.kind(w, m.aNs) ?: ((EcgSnap.valueAt(w, m.aNs) ?: median) >= median) }
            val above = inRow.any { !it.isAmplitude } || sides.any { it }
            val below = sides.any { !it }
            rowLayout(
                w, a, rowNs,
                padTopPt = if (above) LABEL_PAD_PT else 0f,
                padBottomPt = if (below) LABEL_PAD_PT else 0f,
                maxMm = maxMm,
            )
        }
        // Rows packed onto the pages as they fit.
        val pagesOfRows = mutableListOf<MutableList<Int>>()
        var y = Float.MAX_VALUE
        for (r in 0 until rows) {
            val h = layouts[r].blockPt
            if (pagesOfRows.isEmpty() || y + h > sheet.pageH - FOOTER_ROOM_PT) {
                pagesOfRows += mutableListOf<Int>()
                y = rowsTop
            }
            pagesOfRows.last() += r
            y += h + ROW_GAP_PT
        }
        if (pagesOfRows.isEmpty()) pagesOfRows += mutableListOf<Int>()
        val pages = pagesOfRows.size

        val doc = PdfDocument()
        try {
            for ((pg, rowsOnPage) in pagesOfRows.withIndex()) {
                val page = doc.startPage(PdfDocument.PageInfo.Builder(sheet.pageW, sheet.pageH, pg + 1).create())
                val c = page.canvas
                header(c, p, sheet, title, comment, info)
                scaleLegend(c, p.small, sheet)
                var top = rowsTop
                for (r in rowsOnPage) {
                    val layout = layouts[r]
                    val rowStart = fromNs + r * rowNs
                    // bpm at the start of the strip (first row, left) and at the end of each row with a
                    // marker (right), on the label line above the row, never on the signal.
                    val bpm = "bpm: " + (rowBpm[rowStart]?.toString() ?: "–")
                    val hasMarker = markers.any { it in rowStart until rowStart + rowNs }
                    drawRow(
                        c, w, sheet, rowStart, top, layout, markers, zone, p.small, measurements,
                        bpmLeft = if (r == 0) bpm else null, bpmRight = if (hasMarker) bpm else null,
                    )
                    top += layout.blockPt + ROW_GAP_PT
                }
                val pageNo = Paint(p.small).apply { textAlign = Paint.Align.RIGHT }
                c.drawText("Page ${pg + 1} of $pages", sheet.pageW - MARGIN, sheet.pageH - 8f, pageNo)
                doc.finishPage(page)
            }
            return save(context, doc, fileStem)
        } finally {
            doc.close()
        }
    }

    /**
     * One row from [blockTop]: the label line (time, bpm, marker), the room above the grid,
     * the grid with the mV scale in the margin, the signal, the room below.
     */
    private fun drawRow(
        c: Canvas, w: EcgWindow, sheet: Sheet, startNs: Long, blockTop: Float, layout: RowLayout, markers: List<Long>,
        zone: ZoneId, label: Paint, measurements: List<Measurement>, bpmLeft: String?, bpmRight: String?,
    ) {
        val left = sheet.leftPt
        val width = sheet.rowWidthPt
        val height = layout.heightPt
        val endNs = startNs + sheet.rowNs
        val labelY = blockTop + LABEL_LINE_PT - 3f
        val top = blockTop + LABEL_LINE_PT + layout.padTopPt // top of the grid
        val roomTop = top - layout.padTopPt
        val roomBottom = top + height + layout.padBottomPt
        val timeText = CLOCK.format(local(startNs, zone)) + (bpmLeft?.let { "   $it" } ?: "")
        c.drawText(timeText, left, labelY, label)
        val timeEnd = left + label.measureText(timeText)
        if (bpmRight != null) {
            c.drawText(bpmRight, left + width - label.measureText(bpmRight), labelY, label)
        }

        val minor = Paint().apply { color = Color.rgb(246, 201, 201); strokeWidth = 0.25f }
        val major = Paint().apply { color = Color.rgb(229, 154, 154); strokeWidth = 0.6f }
        val mmX = (sheet.rowSeconds * MM_PER_S).toInt()
        for (i in 0..mmX) {
            val x = left + i * PT_PER_MM
            c.drawLine(x, top, x, top + height, if (i % 5 == 0) major else minor)
        }
        for (i in 0..layout.rowMm.toInt()) {
            val y = top + i * PT_PER_MM
            c.drawLine(left, y, left + width, y, if (i % 5 == 0) major else minor)
        }
        // mV scale in the margin, at every thick line: 0 is 0 mV (the H10's zero).
        val scale = Paint(label).apply { textSize = 6f; color = Color.GRAY; textAlign = Paint.Align.RIGHT }
        for (i in 0..layout.rowMm.toInt() step 5) {
            val mv = (layout.upMm - i) / MM_PER_MV
            val text = if (Math.abs(mv) < 1e-3) "0 mV" else String.format(java.util.Locale.ROOT, "%+.1f", mv)
            c.drawText(text, left - 3f, top + i * PT_PER_MM + 2f, scale)
        }

        val baseY = top + layout.upMm * PT_PER_MM
        val i0 = w.t.binarySearch(startNs).let { if (it < 0) -it - 1 else it }
        val i1 = w.t.binarySearch(endNs).let { if (it < 0) -it - 1 else it }
        val median = 0f // raw values: 0 mV is the H10's zero in every row (no baseline subtracted)
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
            c.drawText("MARKER ${CLOCK.format(local(m, zone))}", maxOf(x + 3f, timeEnd + 8f), labelY, red)
        }
        val xOf = { t: Long -> left + (t - startNs) / 1e9f * MM_PER_S * PT_PER_MM }
        val yOf = { t: Long -> EcgSnap.valueAt(w, t)?.let { baseY - (it - median) * MM_PER_MV * PT_PER_MM } ?: baseY }
        for (m in measurements) {
            if (m.isAmplitude) drawAmplitude(c, w, m.aNs, startNs, endNs, roomTop, roomBottom, median, baseY, xOf, label)
            else drawMeasurement(c, m, startNs, endNs, left, width, roomTop, xOf, yOf, label)
        }
    }

    /**
     * The mV of one point (above or below 0 mV (the H10's zero)) as a small black number next to
     * the wave, never on the trace: above a top, below a bottom (a freely placed point: above if
     * above 0 mV), with a light dashed line from just above (or under) the point to it.
     */
    private fun drawAmplitude(
        c: Canvas, w: EcgWindow, tNs: Long, rowStart: Long, rowEnd: Long, roomTop: Float, roomBottom: Float,
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
        val textY = if (above) minOf(yExt - 4f, y - minLead).coerceAtLeast(roomTop + 7f)
        else (maxOf(yExt + 4f, y + minLead) + 6f).coerceAtMost(roomBottom - 1f)
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
        val lineY = ((ys.minOrNull() ?: (top + 20f)) - 7f).coerceAtLeast(top + 10f) // top = top of the room above the grid
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

    /** The comment (at most 2 lines) and the description, wrapped to the page width (at most 7 lines). */
    private fun headerLines(p: Paints, sheet: Sheet, comment: String, info: List<String>): List<String> {
        val maxW = sheet.pageW - 2 * MARGIN
        val commentLines = wrap("Comment: " + comment.ifBlank { "(none)" }, p.text, maxW).let {
            if (it.size > 2) listOf(it[0], it[1].dropLast(1) + "…") else it
        }
        return (commentLines + info.flatMap { wrap(it, p.text, maxW) }).take(7)
    }

    /** Where the header text ends on the page (the rows start below it). */
    private fun headerBottom(p: Paints, sheet: Sheet, comment: String, info: List<String>): Float =
        56f + (headerLines(p, sheet, comment, info).size - 1) * 11.5f + 4f

    /** Title (made smaller if it would run into the scale legend), comment and description. */
    private fun header(c: Canvas, p: Paints, sheet: Sheet, title: String, comment: String, info: List<String>) {
        val titleMaxW = sheet.pageW - MARGIN - LEGEND_W_PT - 10f - MARGIN
        val titlePaint = Paint(p.title)
        while (titlePaint.measureText(title) > titleMaxW && titlePaint.textSize > 9f) titlePaint.textSize -= 0.5f
        c.drawText(title, MARGIN, 40f, titlePaint)
        headerLines(p, sheet, comment, info).forEachIndexed { i, line -> c.drawText(line, MARGIN, 56f + i * 11.5f, p.text) }
    }

    /** Width of the scale legend in the top right corner of ECG pages. */
    private const val LEGEND_W_PT = 160f

    /**
     * One large square of the grid (5 x 5 small squares) in the top right corner, with what it
     * measures: 25 mm/s gives 40 ms per mm, 10 mm/mV gives 0.1 mV per mm.
     */
    private fun scaleLegend(c: Canvas, small: Paint, sheet: Sheet) {
        val mm = PT_PER_MM
        val x0 = sheet.pageW - MARGIN - LEGEND_W_PT
        val y0 = 16f
        val minor = Paint().apply { color = Color.rgb(246, 201, 201); strokeWidth = 0.25f }
        val major = Paint().apply { color = Color.rgb(229, 154, 154); strokeWidth = 0.6f }
        for (i in 0..5) {
            val p = if (i == 0 || i == 5) major else minor
            c.drawLine(x0 + i * mm, y0, x0 + i * mm, y0 + 5 * mm, p)
            c.drawLine(x0, y0 + i * mm, x0 + 5 * mm, y0 + i * mm, p)
        }
        val ink = Paint(small).apply { color = Color.BLACK; textSize = 7f }
        val tx = x0 + 5 * mm + 5f
        c.drawText("1 small square = 40 ms · 0.1 mV", tx, y0 + 5f, ink)
        c.drawText("1 large square = 200 ms · 0.5 mV", tx, y0 + 13f, ink)
        c.drawText("(25 mm/s, 10 mm/mV)", tx, y0 + 21f, ink)
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
