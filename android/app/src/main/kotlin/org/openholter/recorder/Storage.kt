package org.openholter.recorder

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.StatFs
import androidx.core.content.FileProvider
import androidx.documentfile.provider.DocumentFile
import java.io.File
import java.time.LocalDate

object Storage {
    /**
     * App-specific storage: `/sdcard/Android/data/org.openholter.recorder/files/recordings`.
     * No storage permission needed; removed when the app is uninstalled (export first!).
     */
    fun root(context: Context): File {
        val base = context.getExternalFilesDir(null) ?: context.filesDir
        return File(base, "recordings").apply { mkdirs() }
    }

    data class Usage(val usedBytes: Long, val freeBytes: Long)

    fun usage(context: Context): Usage {
        val root = root(context)
        val used = root.walkTopDown().filter { it.isFile }.sumOf { it.length() }
        val free = runCatching { StatFs(root.path).availableBytes }.getOrDefault(0L)
        return Usage(used, free)
    }

    fun filesOfDay(context: Context, day: LocalDate): List<File> =
        File(root(context), day.toString()).listFiles().orEmpty().filter { it.isFile }.sortedBy { it.name }

    /** Share sheet with all files of one day (e.g. to Drive, a laptop via Quick Share, e-mail). */
    fun shareDayIntent(context: Context, day: LocalDate): Intent? {
        val files = filesOfDay(context, day)
        if (files.isEmpty()) return null
        val uris = ArrayList<Uri>(files.map { FileProvider.getUriForFile(context, "${context.packageName}.files", it) })
        val send = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = "text/*"
            putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
            putExtra(Intent.EXTRA_SUBJECT, "open-holter recording $day")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(send, "Share recording $day")
    }

    data class ExportResult(val copied: Int, val skipped: Int, val failed: Int)

    /**
     * Copies all recordings into a folder picked with the system file picker
     * (Storage Access Framework), e.g. a USB stick or a Drive folder. Files that
     * already exist there with the same size are skipped, so running it again
     * only copies what is new or still growing.
     */
    fun exportToTree(context: Context, treeUri: Uri, onProgress: (String) -> Unit = {}): ExportResult {
        val tree = DocumentFile.fromTreeUri(context, treeUri) ?: return ExportResult(0, 0, 1)
        var copied = 0
        var skipped = 0
        var failed = 0
        val root = root(context)
        val days = root.listFiles().orEmpty().filter { it.isDirectory }.sortedBy { it.name }
        for (dayDir in days) {
            val target = tree.findFile(dayDir.name)?.takeIf { it.isDirectory } ?: tree.createDirectory(dayDir.name)
            if (target == null) {
                failed++
                continue
            }
            for (file in dayDir.listFiles().orEmpty().filter { it.isFile }.sortedBy { it.name }) {
                onProgress("${dayDir.name}/${file.name}")
                val existing = target.findFile(file.name)
                if (existing != null && existing.length() == file.length()) {
                    skipped++
                    continue
                }
                try {
                    existing?.delete()
                    val doc = target.createFile(mimeFor(file), file.name) ?: error("cannot create ${file.name}")
                    context.contentResolver.openOutputStream(doc.uri)!!.use { out ->
                        file.inputStream().use { it.copyTo(out, 256 * 1024) }
                    }
                    copied++
                } catch (e: Exception) {
                    failed++
                }
            }
        }
        return ExportResult(copied, skipped, failed)
    }

    // "application/octet-stream" keeps providers from appending another extension.
    private fun mimeFor(file: File) = if (file.extension == "csv") "text/csv" else "application/octet-stream"
}
