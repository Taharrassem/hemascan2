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
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
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

data class Photo(val bmp: Bitmap, val cells: List<Cell> = emptyList())

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val ok = OpenCVLoader.initLocal()
        setContent { MaterialTheme { App(ok) } }
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
    var busy by remember { mutableStateOf(false) }
    var camUri by remember { mutableStateOf<Uri?>(null) }

    val pick = rememberLauncherForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        uris.forEach { u -> loadBitmap(ctx, u)?.let { photos.add(Photo(it)) } }
        if (photos.isNotEmpty()) { cur = photos.lastIndex; sel = null }
    }
    val cam = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) camUri?.let { u -> loadBitmap(ctx, u)?.let { photos.add(Photo(it)); cur = photos.lastIndex; sel = null } }
    }

    Column(Modifier.fillMaxSize().statusBarsPadding().padding(8.dp)) {
        Text("HemaScan AI", style = MaterialTheme.typography.headlineSmall)
        Text(
            "⚠ Version non validée : aide au comptage uniquement, pas de diagnostic. " +
                "La classification est manuelle (aucun modèle IA entraîné n'est encore intégré).",
            style = MaterialTheme.typography.bodySmall, color = Color(0xFFB00020)
        )
        if (!cvOk) Text("OpenCV n'a pas pu s'initialiser.", color = Color.Red)
        TabRow(selectedTabIndex = tab) {
            listOf("Analyse", "Résultats", "Comparaison").forEachIndexed { i, t ->
                Tab(selected = tab == i, onClick = { tab = i }, text = { Text(t) })
            }
        }
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(top = 8.dp)) {
            val p = photos.getOrNull(cur)
            when (tab) {
                0 -> {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = {
                            val f = File(ctx.cacheDir, "cam_${System.currentTimeMillis()}.jpg")
                            val u = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", f)
                            camUri = u; cam.launch(u)
                        }) { Text("Photo") }
                        Button(onClick = { pick.launch("image/*") }) { Text("Importer") }
                    }
                    if (p == null) Text("Prenez ou importez une photo de frottis.", Modifier.padding(top = 16.dp))
                    else {
                        if (photos.size > 1) LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            items(photos.indices.toList()) { i ->
                                FilterChip(selected = i == cur, onClick = { cur = i; sel = null }, label = { Text("Photo ${i + 1}") })
                            }
                        }
                        Viewer(p, show, filter, sel) { sel = it }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(enabled = !busy, onClick = {
                                scope.launch {
                                    busy = true
                                    val nb = withContext(Dispatchers.Default) { Analyzer.enhance(p.bmp) }
                                    photos[cur] = Photo(nb); sel = null; busy = false
                                }
                            }) { Text("Améliorer") }
                            Button(enabled = !busy, onClick = {
                                scope.launch {
                                    busy = true
                                    val cells = withContext(Dispatchers.Default) { Analyzer.detect(p.bmp) }
                                    photos[cur] = p.copy(cells = cells); sel = null; busy = false
                                }
                            }) { Text(if (busy) "…" else "Détecter") }
                        }
                        Row { Text("Annotations", Modifier.weight(1f).padding(top = 12.dp)); Switch(show, { show = it }) }
                        LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            item { FilterChip(selected = filter == null, onClick = { filter = null }, label = { Text("Tout") }) }
                            items(p.cells.map { it.label }.distinct()) { l ->
                                FilterChip(selected = filter == l, onClick = { filter = l }, label = { Text(l) })
                            }
                        }
                        val s = sel?.let { p.cells.getOrNull(it) }
                        if (s == null) Text("Touchez une cellule pour la classer.", style = MaterialTheme.typography.bodySmall)
                        else {
                            Text("Cellule ${sel!! + 1} — ${s.label} — confiance : ${if (s.conf == 0f) "n/a (manuel)" else "%.0f%%".format(s.conf * 100)}")
                            LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                items(LABELS) { l ->
                                    AssistChip(onClick = {
                                        val nc = p.cells.toMutableList(); nc[sel!!] = s.copy(label = l)
                                        photos[cur] = p.copy(cells = nc)
                                    }, label = { Text(l) })
                                }
                            }
                        }
                    }
                }
                1 -> {
                    if (p == null) Text("Aucune analyse.") else {
                        Text("Photo ${cur + 1} — ${p.cells.size} cellules", style = MaterialTheme.typography.titleMedium)
                        CountTable(p.cells)
                        Button(onClick = { exportCsv(ctx, photos) }, Modifier.padding(top = 8.dp)) { Text("Exporter CSV (toutes photos)") }
                    }
                }
                else -> {
                    if (photos.isEmpty()) Text("Aucune photo.") else {
                        photos.forEachIndexed { i, ph ->
                            val unc = ph.cells.count { it.label == Analyzer.UNCERTAIN }
                            Text("Photo ${i + 1} : ${ph.cells.size} cellules, $unc incertaines")
                        }
                        Text("Total global", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 12.dp))
                        CountTable(photos.flatMap { it.cells })
                        Text(
                            "Les cellules d'un champ qui chevauche un autre ne sont pas dédupliquées (non implémenté).",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun CountTable(cells: List<Cell>) {
    val total = max(cells.size, 1)
    cells.groupBy { it.label }.toList().sortedByDescending { it.second.size }.forEach { (l, v) ->
        Row(Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
            Text(l, Modifier.weight(1f)); Text("${v.size}", Modifier.width(48.dp)); Text("%.1f%%".format(100.0 * v.size / total), Modifier.width(64.dp))
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
        Modifier.fillMaxWidth().height(360.dp).clipToBounds().background(Color.Black)
            .onSizeChanged { size = it }
            .transformable(ts)
            .graphicsLayer(scaleX = scale, scaleY = scale, translationX = off.x, translationY = off.y)
            .pointerInput(p, size) {
                detectTapGestures { t ->
                    val (s, ox, oy) = fit()
                    var best: Int? = null; var bd = Float.MAX_VALUE
                    p.cells.forEachIndexed { i, c ->
                        val d = hypot(t.x - (ox + c.x * s), t.y - (oy + c.y * s))
                        if (d <= max(c.r * s, 24f) && d < bd) { bd = d; best = i }
                    }
                    onSel(best)
                }
            }
    ) {
        Image(img, null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
        Canvas(Modifier.fillMaxSize()) {
            if (!show || size == IntSize.Zero) return@Canvas
            val (s, ox, oy) = fit()
            p.cells.forEachIndexed { i, c ->
                if (filter == null || c.label == filter) {
                    drawCircle(
                        if (i == sel) Color.White else colorFor(c.label), c.r * s,
                        Offset(ox + c.x * s, oy + c.y * s), style = Stroke(if (i == sel) 6f else 3f)
                    )
                }
            }
        }
    }
}
