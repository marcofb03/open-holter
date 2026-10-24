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
import org.openholter.core.EcgWindow
import org.openholter.core.Marker
import org.openholter.core.RecordingFiles
import java.io.File
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.ceil

/**
 * PDF of one symptom marker for a doctor: A4 landscape, standard ECG paper
 * scale (25 mm/s, 10 mm/mV, 1 mm / 5 mm grid), 10 s per row, 4 rows per page,
 * as many pages as the window needs. Raw H10 signal, no filtering.
 */
object MarkerPdf {
    private const val PT_PER_MM = 72f / 25.4f
    private const val PAGE_W = 842 // A4 landscape in points
    private const val PAGE_H = 595
    private const val MM_PER_S = 25f
    private const val MM_PER_MV = 10f
    private const val SECONDS_PER_ROW = 10
    private const val ROW_MM = 35f
    private const val ROWS_PER_PAGE = 4
    private const val ROW_GAP_PT = 16f
    private const val TOP_PT = 112f

    private val DATE_TIME: DateTimeFormatter = DateTimeFormatter.ofPattern("EEE d MMM yyyy, HH:mm:ss")
    private val CLOCK: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss")
    private val FILE_STAMP: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")

    fun write(context: Context, marker: Marker, w: EcgWindow, fromNs: Long, toNs: Long, hrBpm: Int?): File {
        val zone = ZoneId.systemDefault()
        val at = RecordingFiles.instantOf(marker.timestampNs).atZone(zone)
        val rowNs = SECONDS_PER_ROW * 1_000_000_000L
        val rows = ceil((toNs - fromNs).toDouble() / rowNs).toInt()
        val pages = maxOf(1, ceil(rows.toDouble() / ROWS_PER_PAGE).toInt())
        val beforeS = (marker.timestampNs - fromNs) / 1_000_000_000L
        val afterS = (toNs - marker.timestampNs) / 1_000_000_000L

        val title = Paint().apply { textSize = 14f; typeface = Typeface.DEFAULT_BOLD; isAntiAlias = true }
        val text = Paint().apply { textSize = 9.5f; isAntiAlias = true }
        val small = Paint().apply { textSize = 8f; color = Color.DKGRAY; isAntiAlias = true }
        // The comment gets at most 2 lines; the description of the recording is always shown.
        val maxW = PAGE_W - 2 * 40f
        val commentLines = wrap("Comment: " + marker.comment.ifBlank { "(none)" }, text, maxW).let {
            if (it.size > 2) listOf(it[0], it[1].dropLast(1) + "…") else it
        }
        val headerLines = commentLines + buildList {
            add(
                "ECG from $beforeS s before to $afterS s after the marker. Polar H10 chest strap, single lead, 130 Hz, " +
                    "raw (not filtered). Scale 25 mm/s, 10 mm/mV." +
                    (hrBpm?.let { " H10 heart rate around the marker: $it bpm." } ?: ""),
            )
            if (w.size == 0) add("No ECG was recorded around this marker (strap not connected or recording stopped).")
        }.flatMap { wrap(it, text, maxW) }

        val doc = PdfDocument()
        try {
            for (p in 0 until pages) {
                val page = doc.startPage(PdfDocument.PageInfo.Builder(PAGE_W, PAGE_H, p + 1).create())
                val c = page.canvas
                c.drawText("Symptom marker – ${DATE_TIME.format(at)} (${zone.id})", 40f, 40f, title)
                headerLines.take(5).forEachIndexed { i, line -> c.drawText(line, 40f, 56f + i * 11.5f, text) }
                for (k in 0 until ROWS_PER_PAGE) {
                    val r = p * ROWS_PER_PAGE + k
                    if (r >= rows) break
                    val top = TOP_PT + k * (ROW_MM * PT_PER_MM + ROW_GAP_PT)
                    drawRow(c, w, fromNs + r * rowNs, top, marker.timestampNs, zone, small)
                }
                c.drawText(
                    "Recorded with open-holter (open source). Not a medical device, not for diagnosis on its own. " +
                        "Print at 100 % (no \"fit to page\") to keep 25 mm/s and 10 mm/mV.   Page ${p + 1} of $pages",
                    40f, PAGE_H - 18f, small,
                )
                doc.finishPage(page)
            }
            val dir = File(context.cacheDir, "reports").apply { mkdirs() }
            dir.listFiles()?.forEach { it.delete() } // only the latest export is kept
            val file = File(dir, "open-holter_marker_${FILE_STAMP.format(at)}.pdf")
            file.outputStream().use { doc.writeTo(it) }
            return file
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
        return Intent.createChooser(send, "Share or save the ECG report")
    }

    private fun drawRow(c: Canvas, w: EcgWindow, startNs: Long, top: Float, markerNs: Long, zone: ZoneId, label: Paint) {
        val left = (PAGE_W - SECONDS_PER_ROW * MM_PER_S * PT_PER_MM) / 2
        val width = SECONDS_PER_ROW * MM_PER_S * PT_PER_MM
        val height = ROW_MM * PT_PER_MM
        val endNs = startNs + SECONDS_PER_ROW * 1_000_000_000L
        c.drawText(CLOCK.format(RecordingFiles.instantOf(startNs).atZone(zone)), left, top - 3f, label)

        val minor = Paint().apply { color = Color.rgb(246, 201, 201); strokeWidth = 0.25f }
        val major = Paint().apply { color = Color.rgb(229, 154, 154); strokeWidth = 0.6f }
        val mmX = (SECONDS_PER_ROW * MM_PER_S).toInt()
        for (i in 0..mmX) {
            val x = left + i * PT_PER_MM
            c.drawLine(x, top, x, top + height, if (i % 5 == 0) major else minor)
        }
        for (i in 0..ROW_MM.toInt()) {
            val y = top + i * PT_PER_MM
            c.drawLine(left, y, left + width, y, if (i % 5 == 0) major else minor)
        }

        val i0 = w.t.binarySearch(startNs).let { if (it < 0) -it - 1 else it }
        val i1 = w.t.binarySearch(endNs).let { if (it < 0) -it - 1 else it }
        if (i1 > i0) {
            val median = w.mv.copyOfRange(i0, i1).sortedArray()[(i1 - i0) / 2]
            val baseY = top + height * 0.6f // room for 2.1 mV up, 1.4 mV down
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
        if (markerNs in startNs until endNs) {
            val x = left + (markerNs - startNs) / 1e9f * MM_PER_S * PT_PER_MM
            val red = Paint().apply { color = Color.rgb(211, 47, 47); strokeWidth = 1.5f; textSize = 8f; isAntiAlias = true }
            c.drawLine(x, top, x, top + height, red)
            c.drawText("MARKER ${CLOCK.format(RecordingFiles.instantOf(markerNs).atZone(zone))}", x + 3f, top + 9f, red)
        }
    }

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
