package org.openholter.recorder

import android.Manifest
import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager
import android.os.VibrationEffect
import android.os.VibratorManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import com.polar.androidcommunications.api.ble.model.gatt.client.pmd.PmdMeasurementType
import com.polar.sdk.api.PolarBleApi
import com.polar.sdk.api.model.EcgSample
import com.polar.sdk.api.model.PolarAccelerometerData
import com.polar.sdk.api.model.PolarEcgData
import com.polar.sdk.api.model.PolarHrData
import com.polar.sdk.api.model.PolarSensorSetting
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.openholter.core.AccCsvWriter
import org.openholter.core.ClockMapper
import org.openholter.core.EcgCsvWriter
import org.openholter.core.EcgPreviewBuffer
import org.openholter.core.EventLog
import org.openholter.core.HrCsvWriter
import org.openholter.core.RecorderMachine
import org.openholter.core.RecorderMachine.Action
import org.openholter.core.RecorderMachine.Event
import org.openholter.core.RecorderMachine.State
import org.openholter.core.RecordingFiles
import java.io.File
import java.time.LocalDate
import java.time.ZoneId

/**
 * Foreground service (type `connectedDevice`) that owns the H10 connection and
 * writes the files. Survives screen-off and app swipe; restarted by the
 * system (START_STICKY) and after reboot / app update ([BootReceiver]).
 *
 * All recording state is touched only on [worker] (one thread at a time).
 * Connection decisions come from [RecorderMachine]; this class just executes
 * its actions against the Polar SDK.
 */
class RecorderService : LifecycleService() {

    @OptIn(ExperimentalCoroutinesApi::class)
    private val worker = Dispatchers.IO.limitedParallelism(1)

    private val app by lazy { HolterApp.from(this) }
    private val api: PolarBleApi by lazy { app.polar }
    private val prefs by lazy { app.prefs }
    private val zone: ZoneId get() = ZoneId.systemDefault()

    private val machine = RecorderMachine()
    private val events = Channel<Event>(Channel.UNLIMITED)

    private var ecgWriter: EcgCsvWriter? = null
    private var accWriter: AccCsvWriter? = null
    private var hrWriter: HrCsvWriter? = null
    private lateinit var eventLog: EventLog
    private val ecgClock = ClockMapper()
    private val accClock = ClockMapper()
    private val preview = EcgPreviewBuffer(RecorderHub.PREVIEW_SAMPLES)

    private var ecgJob: Job? = null
    private var accJob: Job? = null
    private var hrJob: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null

    private var statsDay: LocalDate = LocalDate.now()
    private var statsSamples = 0L
    private var lastEcgEventMs = 0L
    private var battery: Int? = null
    private var heartRate: Int? = null
    private var contact: Boolean? = null
    private var linkSinceMs: Long? = null

    override fun onCreate() {
        super.onCreate()
        createChannel()
        if (!goForeground()) {
            stopSelf()
            return
        }
        eventLog = EventLog(Storage.root(this), zone)
        statsDay = LocalDate.now(zone)
        statsSamples = prefs.ecgSamples(statsDay)
        lifecycleScope.launch(worker) { processEvents() }
        lifecycleScope.launch(worker) { PolarEvents.events.collect { onPolarEvent(it) } }
        lifecycleScope.launch(worker) { timers() }
        lifecycleScope.launch(worker) { publishStorage() }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        val action = intent?.action ?: ACTION_RESTORE // null intent = restarted by the system
        lifecycleScope.launch(worker) { onCommand(action, intent) }
        return START_STICKY
    }

    override fun onBind(intent: Intent): IBinder? {
        super.onBind(intent)
        return null
    }

    override fun onDestroy() {
        // Leave the H10 in a clean state (streams stopped) and close the files.
        runBlocking(worker) {
            withTimeoutOrNull(4_000) {
                val s = machine.state
                if (s is State.Active) {
                    stopPmdStreams(s.deviceId)
                    stopHr(s.deviceId)
                }
            }
            closeWriters()
        }
        releaseWakeLock()
        RecorderHub.status.value = RecorderHub.status.value.copy(serviceRunning = false, link = RecorderHub.Link.IDLE)
        super.onDestroy()
    }

    // ---- commands -------------------------------------------------------------------------

    private fun onCommand(action: String, intent: Intent?) {
        when (action) {
            ACTION_START -> {
                val id = intent?.getStringExtra(EXTRA_DEVICE_ID) ?: prefs.deviceId ?: return
                prefs.deviceId = id
                prefs.recordingActive = true
                prefs.paused = false
                startRecording(id, paused = false)
            }
            ACTION_RESTORE -> {
                val id = prefs.deviceId
                if (prefs.recordingActive && id != null && machine.state == State.Idle) {
                    prefs.updateFrom?.let {
                        log("update_installed", "$it -> ${BuildConfig.VERSION_NAME}; ECG resumes after the update gap")
                        prefs.updateFrom = null
                    }
                    log("service_restart", "restored recording")
                    startRecording(id, paused = prefs.paused)
                }
            }
            ACTION_PAUSE -> {
                prefs.paused = true
                events.trySend(Event.Pause)
            }
            ACTION_RESUME -> {
                prefs.paused = false
                events.trySend(Event.Resume)
            }
            ACTION_MARK -> mark()
            ACTION_UPDATE -> lifecycleScope.launch(worker) { prepareUpdate(intent) }
            ACTION_UPDATE_FAILED -> {
                prefs.updateFrom = null
                log("update_failed", intent?.getStringExtra(EXTRA_DETAIL).orEmpty())
                if (!prefs.paused) events.trySend(Event.Resume)
            }
            ACTION_STOP -> {
                prefs.recordingActive = false
                prefs.paused = false
                events.trySend(Event.Stop)
            }
        }
        // Started for nothing (e.g. a marker while not recording): do not linger.
        if (!prefs.recordingActive && machine.state == State.Idle) finishRecording()
    }

    private fun startRecording(deviceId: String, paused: Boolean) {
        openWriters()
        acquireWakeLock()
        events.trySend(Event.Start(deviceId, paused))
    }

    private fun mark() {
        val now = RecordingFiles.nowNs()
        val writer = ecgWriter
        if (writer == null) {
            Log.w(TAG, "marker ignored: not recording")
            return
        }
        writer.addMarker(now)
        log("marker", "symptom marker")
        val ms = now / 1_000_000
        prefs.lastMarkerMs = ms
        vibrate()
        publishStatus()
    }

    /**
     * Before an app update: log it, stop the H10 streams and disconnect cleanly
     * (as for a pause, so the H10 battery is safe), flush the files, then hand
     * the APK to the installer, which ends this process. The new version
     * restores recording ([BootReceiver]) and logs `update_installed`.
     */
    private suspend fun prepareUpdate(intent: Intent?) {
        val apk = intent?.getStringExtra(EXTRA_APK) ?: return
        val to = intent.getStringExtra(EXTRA_VERSION).orEmpty()
        if (prefs.recordingActive) prefs.updateFrom = BuildConfig.VERSION_NAME
        log("update_start", "${BuildConfig.VERSION_NAME} -> $to; ECG stops for the app update (recording gap, not a heart pause)")
        val s = machine.state
        if (s is State.Active && s !is State.Paused) {
            events.trySend(Event.Pause)
            withTimeoutOrNull(10_000) { while (machine.state !is State.Paused) delay(100) }
            delay(UPDATE_TEARDOWN_MS) // Pause stops the streams (1 s grace), then disconnects
        }
        ecgWriter?.flush(sync = true)
        accWriter?.flush(sync = true)
        hrWriter?.flush(sync = true)
        try {
            Updater.commit(this, File(apk))
        } catch (e: Exception) {
            Updater.onInstallFailed(this, e.toString())
        }
    }

    // ---- event loop -----------------------------------------------------------------------

    private suspend fun processEvents() {
        for (event in events) {
            val actions = machine.handle(event, System.currentTimeMillis())
            for (a in actions) execute(a)
            if (actions.isNotEmpty()) {
                if (machine.state == State.Idle) finishRecording() else publishStatus()
            }
        }
    }

    private suspend fun execute(a: Action) {
        when (a) {
            is Action.Log -> log(a.event, a.detail)
            is Action.Connect -> try {
                api.connectToDevice(a.deviceId)
            } catch (e: Exception) {
                log("connect_error", e.toString())
            }
            is Action.Disconnect -> {
                // Belt and braces for the H10 battery issue: streams always go first.
                stopPmdStreams(a.deviceId)
                stopHr(a.deviceId)
                try {
                    api.disconnectFromDevice(a.deviceId)
                } catch (e: Exception) {
                    log("disconnect_error", e.toString())
                }
            }
            is Action.StartStreams -> startStreams(a.deviceId)
            is Action.StopStreams -> stopPmdStreams(a.deviceId)
        }
    }

    private fun onPolarEvent(e: PolarEvents.E) {
        when (e) {
            is PolarEvents.E.Connecting -> Unit
            is PolarEvents.E.Connected -> {
                linkSinceMs = System.currentTimeMillis()
                events.trySend(Event.DeviceConnected(e.deviceId))
            }
            is PolarEvents.E.Disconnected -> {
                linkSinceMs = System.currentTimeMillis()
                if (e.reason.isNotEmpty()) log("disconnect_reason", e.reason)
                hrJob?.cancel()
                hrJob = null
                heartRate = null
                contact = null
                events.trySend(Event.DeviceDisconnected(e.deviceId))
            }
            is PolarEvents.E.StreamingReady -> events.trySend(Event.StreamingReady(e.deviceId))
            is PolarEvents.E.HrReady -> {
                // HR runs whenever the link is up (also when ECG is stopped for lack of
                // contact) because it reports skin contact.
                val s = machine.state
                if (s is State.Active && s !is State.Paused && s.deviceId == e.deviceId) startHr(e.deviceId)
            }
            is PolarEvents.E.Battery -> {
                battery = e.percent
                log("h10_battery", "${e.percent}%")
                publishStatus()
            }
            is PolarEvents.E.Power -> events.trySend(Event.BluetoothPower(e.on))
        }
    }

    private suspend fun timers() {
        var second = 0L
        while (lifecycleScope.isActive) {
            delay(1_000)
            second++
            val nowNs = RecordingFiles.nowNs()
            ecgWriter?.tick(nowNs)
            accWriter?.tick(nowNs)
            hrWriter?.tick(nowNs)
            if (second % 5 == 0L) events.trySend(Event.Tick)
            if (second % 5 == 0L) {
                RecorderHub.ecgPreview.value = preview.snapshot()
                publishStatus()
            }
            if (second % 60 == 0L) {
                prefs.setEcgSamples(statsDay, statsSamples)
                publishStorage()
            }
        }
    }

    private fun publishStorage() {
        val usage = Storage.usage(this)
        RecorderHub.status.value = RecorderHub.status.value.copy(
            storageUsedBytes = usage.usedBytes, storageFreeBytes = usage.freeBytes,
        )
    }

    // ---- streams --------------------------------------------------------------------------

    private suspend fun startStreams(deviceId: String) {
        stopPmdStreams(deviceId, quiet = true)
        ecgClock.reset()
        accClock.reset()
        if (hrJob?.isActive != true) startHr(deviceId)

        ecgJob = lifecycleScope.launch(worker) {
            try {
                val settings = api.requestStreamSettings(deviceId, PolarBleApi.PolarDeviceDataType.ECG).maxSettings()
                api.startEcgStreaming(deviceId, settings).collect { onEcg(it) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                log("ecg_stream_error", e.toString())
            }
        }
        accJob = lifecycleScope.launch(worker) {
            // Start ACC after ECG so the two control-point commands do not overlap.
            delay(1_500)
            try {
                val settings = accSettings(deviceId)
                log("acc_stream", settings.settings.toString())
                api.startAccStreaming(deviceId, settings).collect { onAcc(it) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                log("acc_stream_error", e.toString())
            }
        }
    }

    private suspend fun accSettings(deviceId: String): PolarSensorSetting {
        val available = api.requestStreamSettings(deviceId, PolarBleApi.PolarDeviceDataType.ACC).settings
        val wanted = prefs.accRateHz
        val chosen = mutableMapOf<PolarSensorSetting.SettingType, Int>()
        for ((type, values) in available) {
            if (values.isEmpty()) continue
            chosen[type] = when (type) {
                PolarSensorSetting.SettingType.SAMPLE_RATE -> values.minByOrNull { Math.abs(it - wanted) }!!
                // 8 g so running / jumping does not clip.
                PolarSensorSetting.SettingType.RANGE -> if (8 in values) 8 else values.max()
                else -> values.max()
            }
        }
        return PolarSensorSetting(chosen)
    }

    /**
     * Stops ECG/ACC on the H10 and waits briefly so the stop command is sent
     * before anything else (in particular a disconnect) happens.
     */
    private suspend fun stopPmdStreams(deviceId: String, quiet: Boolean = false) {
        val hadStreams = ecgJob?.isActive == true || accJob?.isActive == true
        ecgJob?.cancelAndJoin()
        accJob?.cancelAndJoin()
        ecgJob = null
        accJob = null
        if (!hadStreams && quiet) return
        // Cancelling the flows already makes the SDK stop the measurement; the explicit
        // calls make sure, even if a flow had already failed.
        runCatching { api.stopStreaming(deviceId, PmdMeasurementType.ECG) }
        runCatching { api.stopStreaming(deviceId, PmdMeasurementType.ACC) }
        if (hadStreams) delay(STOP_GRACE_MS)
        ecgWriter?.flush()
        accWriter?.flush()
    }

    private fun startHr(deviceId: String) {
        if (hrJob?.isActive == true) return
        hrJob = lifecycleScope.launch(worker) {
            try {
                api.startHrStreaming(deviceId).collect { onHr(it) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                log("hr_stream_error", e.toString())
            }
        }
    }

    private suspend fun stopHr(deviceId: String) {
        hrJob?.cancelAndJoin()
        hrJob = null
        runCatching { api.stopHrStreaming(deviceId) }
    }

    private fun onEcg(data: PolarEcgData) {
        val samples = data.samples.filterIsInstance<EcgSample>()
        if (samples.isEmpty()) return
        val writer = ecgWriter ?: return
        val arrival = RecordingFiles.nowNs()
        ecgClock.onPacket(arrival, samples.last().timeStamp)
        for (s in samples) {
            val t = ecgClock.map(s.timeStamp)
            writer.writeEcg(t, s.voltage)
            preview.add(s.voltage / 1000f)
        }
        countSamples(samples.size)
        val nowMs = arrival / 1_000_000
        if (nowMs - lastEcgEventMs >= 1_000) {
            lastEcgEventMs = nowMs
            events.trySend(Event.EcgData)
            RecorderHub.ecgPreview.value = preview.snapshot()
        }
    }

    private fun onAcc(data: PolarAccelerometerData) {
        val samples = data.samples
        if (samples.isEmpty()) return
        val writer = accWriter ?: return
        accClock.onPacket(RecordingFiles.nowNs(), samples.last().timeStamp)
        for (s in samples) writer.write(accClock.map(s.timeStamp), s.x, s.y, s.z)
    }

    private fun onHr(data: PolarHrData) {
        val now = RecordingFiles.nowNs()
        for (s in data.samples) {
            hrWriter?.write(now, s.hr, s.rrsMs, s.contactStatus, s.contactStatusSupported)
            ecgWriter?.addHeartRate(now, s.hr)
            for (rr in s.rrsMs) ecgWriter?.addRr(now, rr)
            heartRate = s.hr
            contact = if (s.contactStatusSupported) s.contactStatus else null
            events.trySend(Event.Contact(s.contactStatus, s.contactStatusSupported))
        }
    }

    private fun countSamples(n: Int) {
        val today = LocalDate.now(zone)
        if (today != statsDay) {
            prefs.setEcgSamples(statsDay, statsSamples)
            statsDay = today
            statsSamples = prefs.ecgSamples(today)
        }
        statsSamples += n
    }

    // ---- files ----------------------------------------------------------------------------

    private fun openWriters() {
        if (ecgWriter != null) return
        val root = Storage.root(this)
        ecgWriter = EcgCsvWriter(root, zone, onGap = { from, to ->
            log("ecg_gap", "%.1f s without ECG since %d".format((to - from) / 1e9, from))
        })
        accWriter = AccCsvWriter(root, zone)
        hrWriter = HrCsvWriter(root, zone)
    }

    private fun closeWriters() {
        runCatching { ecgWriter?.close() }
        runCatching { accWriter?.close() }
        runCatching { hrWriter?.close() }
        ecgWriter = null
        accWriter = null
        hrWriter = null
        prefs.setEcgSamples(statsDay, statsSamples)
    }

    private fun finishRecording() {
        closeWriters()
        releaseWakeLock()
        RecorderHub.status.value = RecorderHub.status.value.copy(link = RecorderHub.Link.IDLE, heartRate = null)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun log(event: String, detail: String = "") {
        Log.i(TAG, "$event $detail")
        if (::eventLog.isInitialized) runCatching { eventLog.log(RecordingFiles.nowNs(), event, detail) }
    }

    // ---- status & notification -----------------------------------------------------------

    private fun publishStatus() {
        val s = machine.state
        val link = when (s) {
            State.Idle -> RecorderHub.Link.IDLE
            is State.Connecting -> RecorderHub.Link.CONNECTING
            is State.Connected -> RecorderHub.Link.CONNECTED
            is State.Streaming -> RecorderHub.Link.STREAMING
            is State.NoContact -> RecorderHub.Link.NO_CONTACT
            is State.Paused -> RecorderHub.Link.PAUSED
        }
        val status = RecorderHub.status.value.copy(
            serviceRunning = true,
            link = link,
            deviceId = (s as? State.Active)?.deviceId,
            batteryPercent = battery,
            heartRate = if (link == RecorderHub.Link.STREAMING || link == RecorderHub.Link.NO_CONTACT) heartRate else null,
            contact = contact,
            linkSinceMs = linkSinceMs,
            todayRecordedSeconds = statsSamples / 130,
            lastMarkerMs = prefs.lastMarkerMs.takeIf { it > 0 },
            currentFile = ecgWriter?.currentFile?.name,
        )
        RecorderHub.status.value = status
        if (link != RecorderHub.Link.IDLE) {
            getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification(status))
        }
    }

    private fun goForeground(): Boolean {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "BLUETOOTH_CONNECT not granted; cannot run")
            return false
        }
        return try {
            startForeground(
                NOTIFICATION_ID, buildNotification(RecorderHub.status.value.copy(link = RecorderHub.Link.CONNECTING)),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE,
            )
            true
        } catch (e: Exception) {
            Log.e(TAG, "startForeground failed", e)
            false
        }
    }

    private fun createChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Recording", NotificationManager.IMPORTANCE_LOW).apply {
                description = "Shown while the ECG recorder is running"
                setShowBadge(false)
            },
        )
    }

    private fun buildNotification(st: RecorderHub.Status): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val hours = "%.1f h today".format(st.todayRecordedSeconds / 3600.0)
        val title = when (st.link) {
            RecorderHub.Link.STREAMING -> "Recording ECG" + (st.heartRate?.let { " · $it bpm" } ?: "")
            RecorderHub.Link.CONNECTING -> "Waiting for H10…"
            RecorderHub.Link.CONNECTED -> "H10 connected, starting…"
            RecorderHub.Link.NO_CONTACT -> "No skin contact – strap off?"
            RecorderHub.Link.PAUSED -> "Paused – tap Resume when the strap is back on"
            RecorderHub.Link.IDLE -> "Not recording"
        }
        val b = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_heart)
            .setContentTitle(title)
            .setContentText(hours + (st.batteryPercent?.let { " · H10 battery $it%" } ?: ""))
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .addAction(0, "Mark symptom", servicePendingIntent(ACTION_MARK, 1))
        if (st.link == RecorderHub.Link.PAUSED) {
            b.addAction(0, "Resume", servicePendingIntent(ACTION_RESUME, 2))
        } else {
            b.addAction(0, "Pause for shower", servicePendingIntent(ACTION_PAUSE, 3))
        }
        return b.build()
    }

    private fun servicePendingIntent(action: String, requestCode: Int): PendingIntent =
        PendingIntent.getService(
            this, requestCode, Intent(this, RecorderService::class.java).setAction(action),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )

    // ---- power ----------------------------------------------------------------------------

    @SuppressLint("WakelockTimeout")
    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        // Partial wake lock: keeps the CPU (not the screen) awake so the timers that
        // flush files and watch the connection keep running in Doze.
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "openholter:recording")
            .apply { acquire() }
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    private fun vibrate() {
        runCatching {
            getSystemService(VibratorManager::class.java).defaultVibrator
                .vibrate(VibrationEffect.createOneShot(150, VibrationEffect.DEFAULT_AMPLITUDE))
        }
    }

    companion object {
        private const val TAG = "RecorderService"
        private const val CHANNEL_ID = "recording"
        private const val NOTIFICATION_ID = 1
        private const val STOP_GRACE_MS = 1_000L
        private const val UPDATE_TEARDOWN_MS = 2_500L

        const val ACTION_START = "org.openholter.recorder.START"
        const val ACTION_STOP = "org.openholter.recorder.STOP"
        const val ACTION_PAUSE = "org.openholter.recorder.PAUSE"
        const val ACTION_RESUME = "org.openholter.recorder.RESUME"
        const val ACTION_MARK = "org.openholter.recorder.MARK"
        const val ACTION_RESTORE = "org.openholter.recorder.RESTORE"
        const val ACTION_UPDATE = "org.openholter.recorder.UPDATE"
        const val ACTION_UPDATE_FAILED = "org.openholter.recorder.UPDATE_FAILED"
        const val EXTRA_DEVICE_ID = "device_id"
        private const val EXTRA_APK = "apk"
        private const val EXTRA_VERSION = "version"
        private const val EXTRA_DETAIL = "detail"

        fun send(context: Context, action: String, deviceId: String? = null) {
            val intent = Intent(context, RecorderService::class.java).setAction(action)
            if (deviceId != null) intent.putExtra(EXTRA_DEVICE_ID, deviceId)
            ContextCompat.startForegroundService(context, intent)
        }

        fun sendUpdate(context: Context, apk: File, version: String) {
            val intent = Intent(context, RecorderService::class.java).setAction(ACTION_UPDATE)
                .putExtra(EXTRA_APK, apk.absolutePath).putExtra(EXTRA_VERSION, version)
            ContextCompat.startForegroundService(context, intent)
        }

        fun sendUpdateFailed(context: Context, detail: String) {
            val intent = Intent(context, RecorderService::class.java).setAction(ACTION_UPDATE_FAILED)
                .putExtra(EXTRA_DETAIL, detail)
            ContextCompat.startForegroundService(context, intent)
        }
    }
}
