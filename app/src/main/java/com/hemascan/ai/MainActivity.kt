package com.hemascan.ai

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

val LABELS = listOf(
    Analyzer.UNCLASSIFIED, Analyzer.UNCERTAIN,
    "Érythrocyte normal", "Schizocyte", "Sphérocyte", "Elliptocyte", "Ovalocyte", "Microcyte", "Macrocyte",
    "Codocyte", "Dacryocyte", "Drépanocyte", "Acanthocyte", "Échinocyte", "Stomatocyte",
    "Neutrophile", "Lymphocyte", "Monocyte", "Éosinophile", "Basophile", "Plaquette", "Agrégat plaquettaire"
)

val Teal = Color(0xFF0B5C6B)
data class Photo(val bmp: Bitmap, val cells: List<Cell> = emptyList())

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val ok = OpenCVLoader.initLocal()
        setContent {
            MaterialTheme(colorScheme = lightColorScheme(primary = Teal, secondary = Color(0xFF8E2C48))) { App(ok) }
        }
    }
}

fun loadBitmap(ctx: Context, uri: Uri): Bitmap? = try {
    val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, o) }
    var ss = 1
    while (max(o.outWidth, o.outHeight) / ss > 1800) ss *= 2
    val o2 = BitmapFactory.Options().apply { inSampleSize = ss }
    ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, o2) }
} catch (e: Exception) { null }

fun exportCsv(ctx: Context, photos: List<Photo>) {
    val sb = StringBuilder("photo;classe;x;y;rayon_px\n")
    photos.forEachIndexed { i, p -> p.cells.forEach { sb.append("${i + 1};${it.label};${it.x.toInt()};${it.y.toInt()};${it.r.toInt()}\n") } }
    val f = File(ctx.cacheDir, "hemascan.csv"); f.writeText(sb.toString())
    val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", f)
    val i = Intent(Intent.ACTION_SEND).apply {
        type = "text/csv"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(Intent.createChooser(i, "Exporter CSV"))
}

fun colorFor(label: String) = Color.hsv((label.hashCode() and 0x7fffffff) % 360f, 0.8f, 1f)

@Composable
fun App(cvOk: Boolean) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    val photos = remember { mutableStateListOf<Photo>() }
    var cur by remember { mutableStateOf(0) }
    var sel by remember { mutableStateOf<Int?>(null) }
    var filter by remember { mutableStateOf<String?>(null) }
    var show by remember { mutableStateOf(true) }
    var tab by remember { mutableStateOf(0) }
    var busy by remember { mutableStateOf(0) }
    var sens by remember { mutableStateOf(1f) }
    var camUri by remember { mutableStateOf<Uri?>(null) }

    fun run(idx: Int) {
        scope.launch {
            busy += 1
            val bmp = photos[idx].bmp
            val cells = withContext(Dispatchers.Default) {
                try { Analyzer.detect(bmp, sens) } catch (e: Throwable) { emptyList<Cell>() }
            }
            if (idx < photos.size) photos[idx] = photos[idx].copy(cells = cells)
            sel = null; busy -= 1
        }
    }
    fun add(b: Bitmap) {
        photos.add(Photo(b)); cur = photos.lastIndex; sel = null; filter = null; run(photos.lastIndex)
    }

    val pick = rememberLauncherForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        uris.forEach { u -> loadBitmap(ctx, u)?.let { add(it) } }
    }
    val cam = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) camUri?.let { u -> loadBitmap(ctx, u)?.let { add(it) } }
    }

    Scaffold(
        topBar = {
            Surface(color = Teal) {
                Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp)) {
                    Text("HemaScan AI", color = Color.White, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text("Aide au comptage · version non validée · pas de diagnostic", color = Color(0xCCFFFFFF), style = MaterialTheme.typography.labelSmall)
                }
            }
        },
        bottomBar = {
            NavigationBar {
                listOf("🔬" to "Analyse", "📊" to "Résultats", "⚖️" to "Comparer").forEachIndexed { i, (ic, t) ->
                    NavigationBarItem(selected = tab == i, onClick = { tab = i }, icon = { Text(ic) }, label = { Text(t) })
                }
            }
        }
    ) { pad ->
        Column(Modifier.padding(pad).fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (!cvOk) Text("OpenCV n'a pas pu s'initialiser.", color = Color.Red)
            val p = photos.getOrNull(cur)
            when (tab) {
                0 -> {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = {
                            val f = File(ctx.cacheDir, "cam_${System.currentTimeMillis()}.jpg")
                            val u = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", f)
                            camUri = u; cam.launch(u)
                        }, Modifier.weight(1f)) { Text("📷 Photo") }
                        Button(onClick = { pick.launch("image/*") }, Modifier.weight(1f)) { Text("🖼️ Importer") }
                    }
                    if (p == null) {
                        Card(Modifier.fillMaxWidth()) {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Text("Bienvenue", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                                Text("Prenez ou importez une photo de frottis sanguin. La détection démarre toute seule.")
                                Text("Conseils : champ bien éclairé et net, cellules au centre, peu de zones floues.", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    } else {
                        if (photos.size > 1) LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            items(photos.indices.toList()) { i ->
                                FilterChip(selected = i == cur, onClick = { cur = i; sel = null; filter = null }, label = { Text("Photo ${i + 1}") })
                            }
                        }
                        val unc = p.cells.count { it.label == Analyzer.UNCERTAIN }
                        Text(
                            if (busy > 0) "Analyse en cours…" else "${p.cells.size} cellules · $unc incertaines",
                            style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold
                        )
                        if (busy > 0) LinearProgressIndicator(Modifier.fillMaxWidth())
                        Viewer(p, show, filter, sel) { sel = it }
                        Text("Pincez pour zoomer, double-tap pour réinitialiser, touchez une cellule pour la classer.", style = MaterialTheme.typography.labelSmall)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(enabled = busy == 0, onClick = {
                                scope.launch {
                                    busy += 1
                                    val nb = withContext(Dispatchers.Default) { Analyzer.enhance(p.bmp) }
                                    photos[cur] = Photo(nb); busy -= 1; run(cur)
                                }
                            }, Modifier.weight(1f)) { Text("✨ Améliorer") }
                            Button(enabled = busy == 0, onClick = { run(cur) }, Modifier.weight(1f)) { Text("🔄 Re-détecter") }
                        }
                        Text("Sensibilité de détection", style = MaterialTheme.typography.labelLarge)
                        Slider(value = sens, onValueChange = { sens = it }, valueRange = 0.5f..1.5f, onValueChangeFinished = { run(cur) })
                        Row { Text("Afficher les annotations", Modifier.weight(1f)); Switch(show, { show = it }) }
                        LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            item { FilterChip(selected = filter == null, onClick = { filter = null }, label = { Text("Tout") }) }
                            items(p.cells.map { it.label }.distinct()) { l ->
                                FilterChip(selected = filter == l, onClick = { filter = l }, label = { Text(l) })
                            }
                        }
                        val s = sel?.let { p.cells.getOrNull(it) }
                        if (s != null) Card(Modifier.fillMaxWidth()) {
                            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                val conf = when {
                                    s.conf >= 1f -> "classée manuellement"
                                    s.conf == 0f -> "non classée"
                                    else -> "suggestion ~%.0f%% (heuristique)".format(s.conf * 100)
                                }
                                Text("Cellule ${sel!! + 1} : ${s.label}", fontWeight = FontWeight.Bold)
                                Text(conf, style = MaterialTheme.typography.bodySmall)
                                LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                    items(LABELS) { l ->
                                        AssistChip(onClick = {
                                            val nc = p.cells.toMutableList(); nc[sel!!] = s.copy(label = l, conf = 1f)
                                            photos[cur] = p.copy(cells = nc)
                                        }, label = { Text(l) })
                                    }
                                }
                            }
                        }
                    }
                }
                1 -> {
                    if (p == null) Text("Aucune analyse pour l'instant.") else {
                        Text("Photo ${cur + 1} : ${p.cells.size} cellules", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        Bars(p.cells)
                        Button(onClick = { exportCsv(ctx, photos) }, Modifier.fillMaxWidth()) { Text("📤 Exporter en CSV (toutes les photos)") }
                    }
                }
                else -> {
                    if (photos.isEmpty()) Text("Aucune photo.") else {
                        val maxN = max(1, photos.maxOf { it.cells.size })
                        Text("Par photo", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        photos.forEachIndexed { i, ph ->
                            val unc = ph.cells.count { it.label == Analyzer.UNCERTAIN }
                            Column {
                                Text("Photo ${i + 1} : ${ph.cells.size} cellules, $unc incertaines")
                                Box(Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)).background(Color(0x22000000))) {
                                    Box(Modifier.fillMaxWidth(ph.cells.size / maxN.toFloat()).fillMaxHeight().background(Teal))
                                }
                            }
                        }
                        Text("Total global", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        Bars(photos.flatMap { it.cells })
                        Text("Les cellules des champs qui se chevauchent ne sont pas dédupliquées.", style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
        }
    }
}

@Composable
fun Bars(cells: List<Cell>) {
    val total = max(cells.size, 1)
    cells.groupBy { it.label }.toList().sortedByDescending { it.second.size }.forEach { (l, v) ->
        val pct = "%.1f".format(100.0 * v.size / total)
        Column(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
            Row { Text(l, Modifier.weight(1f)); Text("${v.size} · $pct %") }
            Box(Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)).background(Color(0x22000000))) {
                Box(Modifier.fillMaxWidth(v.size / total.toFloat()).fillMaxHeight().background(colorFor(l)))
            }
        }
    }
}

@Composable
fun Viewer(p: Photo, show: Boolean, filter: String?, sel: Int?, onSel: (Int?) -> Unit) {
    var scale by remember { mutableStateOf(1f) }
    var off by remember { mutableStateOf(Offset.Zero) }
    var size by remember { mutableStateOf(IntSize.Zero) }
    val ts = rememberTransformableState { z, pan, _ -> scale = (scale * z).coerceIn(1f, 8f); off += pan }
    val img = remember(p.bmp) { p.bmp.asImageBitmap() }
    val bw = p.bmp.width.toFloat(); val bh = p.bmp.height.toFloat()
    fun fit(): Triple<Float, Float, Float> {
        val s = min(size.width / bw, size.height / bh)
        return Triple(s, (size.width - bw * s) / 2f, (size.height - bh * s) / 2f)
    }
    Box(
        Modifier.fillMaxWidth().height(380.dp).clip(RoundedCornerShape(12.dp)).clipToBounds().background(Color.Black)
            .onSizeChanged { size = it }
            .transformable(ts)
            .graphicsLayer(scaleX = scale, scaleY = scale, translationX = off.x, translationY = off.y)
            .pointerInput(p, size) {
                detectTapGestures(
                    onDoubleTap = { scale = 1f; off = Offset.Zero },
                    onTap = { t ->
                        val (s, ox, oy) = fit()
                        var best: Int? = null; var bd = Float.MAX_VALUE
                        p.cells.forEachIndexed { i, c ->
                            val d = hypot(t.x - (ox + c.x * s), t.y - (oy + c.y * s))
                            if (d <= max(c.r * s, 24f) && d < bd) { bd = d; best = i }
                        }
                        onSel(best)
                    }
                )
            }
    ) {
        Image(img, null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
        Canvas(Modifier.fillMaxSize()) {
            if (!show || size == IntSize.Zero) return@Canvas
            val (s, ox, oy) = fit()
            val k = 1f / scale
            p.cells.forEachIndexed { i, c ->
                if (filter == null || c.label == filter) {
                    drawCircle(
                        if (i == sel) Color.White else colorFor(c.label), c.r * s,
                        Offset(ox + c.x * s, oy + c.y * s), style = Stroke((if (i == sel) 5f else 2.5f) * k)
                    )
                }
            }
        }
    }
}
