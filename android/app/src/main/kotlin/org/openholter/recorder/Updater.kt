package org.openholter.recorder

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Updates the app from the rolling GitHub pre-release "nightly", which CI
 * rebuilds on every push to main (signed with the release key).
 *
 * Only the user starts an update (Check → Update now). The APK's SHA-256 is
 * checked against version.json, and Android itself refuses an APK that is not
 * signed with the same key as the installed app. While recording, the service
 * first stops the H10 streams cleanly and logs `update_start`; the new version
 * logs `update_installed` when it restarts. The ECG gap in between is a
 * recording gap, not a heart pause.
 *
 * This is the only network access of the app: it talks to github.com and
 * GitHub's download host, and it never uploads anything.
 */
object Updater {
    const val BASE_URL = "https://github.com/marcofb03/open-holter/releases/download/nightly/"

    data class Release(
        val versionCode: Int,
        val versionName: String,
        val commit: String,
        val apk: String,
        val sha256: String,
        val size: Long,
    )

    sealed interface State {
        data object Idle : State
        data object Checking : State
        data class UpToDate(val latest: Release) : State
        data class Available(val release: Release) : State
        data class Downloading(val release: Release, val percent: Int) : State
        data class Installing(val release: Release) : State
        data class Failed(val message: String) : State
    }

    /** Outlives the screens, so leaving the About tab does not cancel a download. */
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> get() = _state

    /** Reads version.json and compares it with the installed versionCode. */
    suspend fun check() {
        if (_state.value is State.Downloading || _state.value is State.Installing) return
        _state.value = State.Checking
        _state.value = try {
            val r = withContext(Dispatchers.IO) { fetchRelease() }
            if (r.versionCode > BuildConfig.VERSION_CODE) State.Available(r) else State.UpToDate(r)
        } catch (e: Exception) {
            Log.w(TAG, "update check failed", e)
            State.Failed("Could not check for updates: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    /**
     * Downloads and verifies the APK, then installs it: through the recorder
     * service when it runs (so the H10 streams are stopped first), otherwise
     * directly.
     */
    suspend fun downloadAndInstall(context: Context, release: Release) {
        val app = context.applicationContext
        try {
            val apk = withContext(Dispatchers.IO) { download(app, release) }
            _state.value = State.Installing(release)
            if (RecorderHub.status.value.serviceRunning) {
                RecorderService.sendUpdate(app, apk, release.versionName)
            } else {
                withContext(Dispatchers.IO) { commit(app, apk) }
            }
        } catch (e: Exception) {
            Log.w(TAG, "update failed", e)
            _state.value = State.Failed("Update failed: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    /** Hands the APK to the system installer. On success Android stops this process. */
    fun commit(context: Context, apk: File) {
        val installer = context.packageManager.packageInstaller
        val params = PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL).apply {
            setAppPackageName(context.packageName)
            setSize(apk.length())
            // No confirmation dialog when Android allows it (app updating itself, see
            // PackageInstaller.SessionParams#setRequireUserAction); otherwise the
            // system asks once (STATUS_PENDING_USER_ACTION).
            setRequireUserAction(PackageInstaller.SessionParams.USER_ACTION_NOT_REQUIRED)
        }
        val id = installer.createSession(params)
        installer.openSession(id).use { session ->
            session.openWrite("base.apk", 0, apk.length()).use { out ->
                apk.inputStream().use { it.copyTo(out) }
                session.fsync(out)
            }
            val status = PendingIntent.getBroadcast(
                context, id, Intent(context, UpdateStatusReceiver::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE,
            )
            session.commit(status.intentSender)
        }
    }

    internal fun onInstallFailed(context: Context, message: String) {
        _state.value = State.Failed("Update not installed: $message")
        if (RecorderHub.status.value.serviceRunning) RecorderService.sendUpdateFailed(context, message)
    }

    private fun fetchRelease(): Release {
        val json = JSONObject(httpGet(BASE_URL + "version.json").use { it.readBytes() }.decodeToString())
        return Release(
            versionCode = json.getInt("versionCode"),
            versionName = json.getString("versionName"),
            commit = json.optString("commit"),
            apk = json.getString("apk"),
            sha256 = json.getString("sha256").lowercase(),
            size = json.optLong("size", -1),
        )
    }

    private fun download(context: Context, r: Release): File {
        require(r.apk.matches(Regex("[A-Za-z0-9._-]+\\.apk"))) { "unexpected file name ${r.apk}" }
        val dir = File(context.cacheDir, "update").apply { deleteRecursively(); mkdirs() }
        val file = File(dir, r.apk)
        val digest = MessageDigest.getInstance("SHA-256")
        _state.value = State.Downloading(r, 0)
        httpGet(BASE_URL + r.apk).use { input ->
            file.outputStream().use { out ->
                val buf = ByteArray(64 * 1024)
                var total = 0L
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    digest.update(buf, 0, n)
                    total += n
                    if (r.size > 0) _state.value = State.Downloading(r, (total * 100 / r.size).toInt())
                }
            }
        }
        val sha = digest.digest().joinToString("") { "%02x".format(it) }
        check(sha == r.sha256) { "checksum mismatch, download again" }
        return file
    }

    private fun httpGet(url: String): java.io.InputStream {
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 15_000
        c.readTimeout = 30_000
        c.instanceFollowRedirects = true // github.com → GitHub's download host (both https)
        c.setRequestProperty("Cache-Control", "no-cache")
        if (c.responseCode != HttpURLConnection.HTTP_OK) {
            val code = c.responseCode
            c.disconnect()
            error("HTTP $code for $url")
        }
        return c.inputStream
    }

    private const val TAG = "Updater"
}

/** Result of [Updater.commit]. */
class UpdateStatusReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val status = intent.getIntExtra(PackageInstaller.EXTRA_STATUS, PackageInstaller.STATUS_FAILURE)
        when (status) {
            PackageInstaller.STATUS_PENDING_USER_ACTION -> {
                // Android wants one confirmation tap. The user just pressed "Update now",
                // so the app is in the foreground and may show the system dialog.
                @Suppress("DEPRECATION")
                val confirm = intent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT) ?: return
                context.startActivity(confirm.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            }
            PackageInstaller.STATUS_SUCCESS -> Unit // the new version takes over
            else -> {
                val msg = intent.getStringExtra(PackageInstaller.EXTRA_STATUS_MESSAGE) ?: "status $status"
                Log.w("Updater", "install failed: $msg")
                Updater.onInstallFailed(context, msg)
            }
        }
    }
}
