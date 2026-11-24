param([string]$RenderDirectory)
$ErrorActionPreference='Stop'
$out=$PSScriptRoot
$root=(Resolve-Path "$out/../..").Path
$base='https://github.com/aksarwar8/FaceFind'
$doc="$base/blob/main"
$data=Get-Content "$root/docs/research/benchmark-aggregates.json" -Raw | ConvertFrom-Json
$det=$data.widerface_retinaface_full500_seed42.detectors
$demo=$data.evaluation_studio_demo
function Color($hex){[Convert]::ToInt32($hex.Substring(0,2),16)+256*[Convert]::ToInt32($hex.Substring(2,2),16)+65536*[Convert]::ToInt32($hex.Substring(4,2),16)}
$navy=Color '0B1120';$white=Color 'F5F8FF';$muted=Color 'BECDE0';$cyan=Color '5BD9ED';$blue=Color '568DF7';$faint=Color '283B56';$surface=Color '152238'
$app=New-Object -ComObject PowerPoint.Application
$p=$app.Presentations.Add(0)
$p.PageSetup.SlideWidth=960
$p.PageSetup.SlideHeight=540
$script:n=0
$script:checks=@()
function Rect($x,$y,$w,$h,$fill){
 $a=$s.Shapes.AddShape(1,$x,$y,$w,$h)
 $a.Line.Visible=0
 $a.Fill.ForeColor.RGB=$fill
}
function Text($x,$y,$w,$h,$words,$size=21,$color=$white,$bold=$false,$url=''){
 $q=$s.Shapes.AddTextbox(1,$x,$y,$w,$h)
 $tf=$q.TextFrame
 $tf.MarginLeft=0;$tf.MarginRight=0;$tf.MarginTop=0;$tf.MarginBottom=0
 $tf.WordWrap=-1;$tf.AutoSize=0
 $tr=$tf.TextRange
 $tr.Text=$words.Replace('|',[string][char]13)
 $tr.Font.Name='Arial';$tr.Font.Size=$size;$tr.Font.Bold=(-1*[int]$bold);$tr.Font.Color.RGB=$color
 if($url){$q.ActionSettings.Item(1).Hyperlink.Address=$url}
 $script:checks+=@([pscustomobject]@{Slide=$script:n;Shape=$q;Text=$words;Width=$w;Height=$h})
}
function Rule($y){Rect 44 $y 872 1 $faint}
function Slide($title,$topic,$ref,$url){
 $script:n++
 $script:s=$p.Slides.Add($script:n,12)
 $s.FollowMasterBackground=0
 $s.Background.Fill.ForeColor.RGB=$navy
 Rect 44 32 30 3 $cyan
 Text 84 26 830 20 $topic 12 $cyan $true
 if($title){Text 44 59 870 56 $title 34 $white $true}
 Rule 500
 Text 44 512 117 20 'FaceFind' 12 $cyan $true
 Text 173 511 705 20 $ref 11 $muted $false $url
 Text 891 510 37 20 ('{0:00}' -f $script:n) 12 $white $true
}
function Image($name,$x,$y,$w,$maxH){
 $pic=$s.Shapes.AddPicture("$root/docs/screenshots/$name.png",0,-1,$x,$y,-1,-1)
 $pic.LockAspectRatio=-1;$pic.Width=$w
 if($pic.Height -gt $maxH){$pic.Height=$maxH}
}
function Bullets($x,$y,$w,$items,$step=53,$size=19){
 foreach($item in $items){
  Rect $x ($y+7) 5 5 $cyan
  Text ($x+16) $y ($w-16) ($step-3) $item $size $muted
  $y+=$step
 }
}
function Head($x,$y,$w,$words){Text $x $y $w 33 $words 21 $cyan $true}
function Node($x,$y,$w,$h,$words){
 Rect $x $y $w $h $surface
 Rect $x $y 3 $h $blue
 Text ($x+11) ($y+9) ($w-19) ($h-13) $words 16 $white $true
}
function Arrow($x1,$y1,$x2,$y2){
 $a=$s.Shapes.AddLine($x1,$y1,$x2,$y2)
 $a.Line.ForeColor.RGB=$cyan;$a.Line.Weight=1.6;$a.Line.EndArrowheadStyle=3
}
function Table($x,$y,$widths,$rows,$size=13,$height=22){
 $all=($widths|Measure-Object -Sum).Sum
 for($r=0;$r -lt $rows.Count;$r++){
  $yy=$y+$r*$height
  if($r -eq 0){Rect $x $yy $all $height $surface}
  Rect $x ($yy+$height-1) $all 1 $faint
  $xx=$x
  for($j=0;$j -lt $widths.Count;$j++){
   $col=if($r -eq 0){$cyan}else{$white}
   Text ($xx+6) ($yy+3) ($widths[$j]-10) ($height-4) $rows[$r][$j] $size $col ($r -eq 0)
   $xx+=$widths[$j]
  }
 }
}
function Fmt($n,$pattern){$n.ToString($pattern,[Globalization.CultureInfo]::InvariantCulture)}
try {
 Slide '' 'INDUSTRY.AI MACHINE LEARNING INTERNSHIP / TASK A' 'Public project repository' $base
 Text 44 95 420 84 'FaceFind' 66 $white $true
 Text 46 192 410 110 'Offline Open-Set Face|Identification for|Group Photographs' 29 $white $true
 Text 46 315 395 67 'Identify enrolled people in a group photo and label unmatched faces Unknown on a local CPU.' 19 $muted
 Text 46 404 410 26 'Sarwar Karim Akhand' 20 $cyan $true
 Text 46 433 405 26 'IIT Madras / BS Data Science and Applications' 16 $white
 Text 46 463 405 22 'github.com/aksarwar8/FaceFind' 16 $cyan $false $base
 Image 'dashboard' 469 146 447 255
 Text 470 382 445 27 'FaceFind Dashboard, with private content masked' 12 $muted

 Slide 'Understanding the problem' 'TASK A / WHO IS IN THE PHOTO?' 'Source  README workflow' "$doc/README.md#application-workflow"
 Text 44 125 845 62 'Give a non-technical operator a browser workflow for local CPU identification.' 24 $muted
 Node 44 204 247 76 'Reference folder|Approximately 50 named identities'
 Node 355 204 244 76 'Unseen group photo|Detect and compare faces'
 Node 663 204 253 76 'Results|Named identities + Unknown'
 Arrow 301 242 343 242
 Arrow 609 242 651 242
 Head 44 333 405 'Closed-set classification'
 Text 44 371 392 75 'A classifier assigns a label from its fixed set of classes.' 22 $muted
 Head 480 333 425 'Open-set identification'
 Text 480 371 426 100 'A gallery can change. Weak or ambiguous matches can remain Unknown.' 22 $muted

 Slide 'Research and model selection' 'MODEL CHOICE' 'Sources  Research report and checkpoint rights' "$doc/RESEARCH.md"
 Head 44 126 420 'Detection'
 Bullets 44 164 420 @('YuNet was compact and fast on CPU.','SCRFD improved precision and recall on the shared 500-image sample.','RetinaFace MobileNet was a tested alternative. The YOLOv8n-Face comparison was partial.') 61 18
 Head 487 126 429 'Recognition'
 Bullets 487 164 429 @('SFace supports a small, bundled CPU installation.','ArcFace had stronger observed unknown rejection in the recognition-only cohort.','AdaFace tied ArcFace on known acceptance, at 55.6 vs 789.2 ms per embedding.') 61 18
 Rule 361
 Text 44 377 421 81 'Preferred: SCRFD + AdaFace|113.03 MB combined. Authorized local weights required.' 19 $white
 Text 487 377 428 81 'Bundled fallback: YuNet + SFace|38.93 MB combined. Lower CPU latency and automatic startup recovery.' 19 $white
 Text 44 469 870 24 'SCRFD pretrained use is research-restricted. AdaFace checkpoint rights remain unresolved.' 13 $muted

 Slide 'System architecture' 'LOCAL BROWSER + FASTAPI ON ONE WORKSTATION' 'Open genuine Archify output and editable specification' "$doc/ARCHITECTURE.md#archify-artifact"
 Node 44 130 175 65 'Browser|Upload and review'
 Node 255 130 190 65 'FastAPI|Validate + preprocess'
 Node 481 130 188 65 'Detect and align|Five landmarks'
 Node 705 130 211 65 'Embed on local CPU|Normalize vectors'
 Arrow 224 161 247 161
 Arrow 450 161 474 161
 Arrow 674 161 698 161
 Arrow 811 197 811 219
 Node 705 225 211 84 'Reference enrollment|Pipeline-local gallery|In-memory vectors'
 Node 440 225 213 84 'Group-photo matching|Cosine, threshold and|top-two margin'
 Node 162 225 229 84 'Resolve duplicate claims|Annotate JPEG and emit|structured results'
 Arrow 699 269 660 269
 Arrow 432 269 397 269
 Arrow 705 190 656 222
 Head 44 333 870 'Independent evaluation in the browser'
 Node 44 380 215 65 'Frozen inference|+ independent truth'
 Node 298 380 249 65 'Evaluation Studio|IoU pairing or manual review'
 Node 584 380 332 65 'Verified outcomes and metrics|Separate evaluation exports'
 Arrow 266 410 291 410
 Arrow 553 410 578 410

 Text 44 465 865 25 'Presentation view derived from the Archify specification and the documented inference flow.' 13 $muted

 Slide 'Engineering investigations' 'FAILURES THAT CHANGED THE IMPLEMENTATION' 'Sources  Research diagnostics and Evaluation Studio' "$doc/RESEARCH.md"
 $items=@(
  @('Scale-sensitive reference misses','Three references returned no valid SCRFD boxes in the 1,280 px pass.','A 640 px gallery-only retry recovered all three. Coverage rose from 14/17 to 17/17 with the same score and NMS settings.','This covers the original folder. Group-photo analysis keeps one pass.'),
  @('False positive and enrollment intent','A brick-wall recess scored 0.5330. Zero overlap with the true face made suppression by NMS impossible.','Offer dominant-face selection with strict review and manual correction.','Smart chose the wrong face in all three smaller-target cases where it auto-selected.'),
  @('Prediction counts and correctness','A count of recognized faces leaves wrong identities and missed faces unresolved.','Freeze inference and compare it with independent ground truth in Evaluation Studio.','With no supplied truth, correctness remains unscored.')
 )
 for($k=0;$k -lt $items.Count;$k++){
  $y=119+123*$k
  Head 44 $y 873 $items[$k][0]
  Text 44 ($y+28) 280 87 $items[$k][1] 17 $muted
  Text 342 ($y+28) 302 88 $items[$k][2] 17 $white
  Text 662 ($y+28) 254 88 $items[$k][3] 16 $muted
  if($k -lt 2){Rule ($y+117)}
 }

 Slide 'Experimental results' 'THREE DISTINCT EVIDENCE COHORTS' 'Sources  Sanitized benchmark aggregates and release verification' "$doc/docs/research/EVIDENCE.md"
 Head 44 121 871 'Controlled synthetic demo / 60 references and five composites'
 $nk=[int]$demo.actual_known;$nu=[int]$demo.actual_unknown
 $ck=[int]$demo.counts.correct_known;$cu=[int]$demo.counts.correct_unknown
 $nums=@(($ck+$cu),$ck,$cu);$den=@(($nk+$nu),$nk,$nu)
 $labels=@('Overall correct outcomes','Known faces correctly named','Unknown faces correctly rejected')
 for($i=0;$i -lt 3;$i++){
  $x=44+$i*291
  Text $x 163 280 55 ((Fmt (100*$nums[$i]/$den[$i]) '0.00').TrimEnd('0').TrimEnd('.')+'%') 42 $white $true
  Text $x 222 280 34 ("{0}/{1}  {2}" -f $nums[$i],$den[$i],$labels[$i]) 15 $muted
 }
 Text 44 256 869 35 'Approximate boxes and calibration overlap limit generalization. Detection TP / FP / FN = 57 / 0 / 0.' 14 $muted
 Rule 290
 Head 44 306 490 'Custom F1 / 500 WIDER FACE images'
 Head 553 306 366 'Warm API mean / one group photo'
 $keys=@('yunet','scrfd_10g_kps','retinaface_mobilenet025');$names=@('YuNet','SCRFD','RetinaFace')
 for($i=0;$i -lt 3;$i++){
  $v=[double]$det.($keys[$i]).custom_metrics.f1;$y=346+29*$i
  Text 44 $y 103 24 $names[$i] 16 $white
  Rect 149 ($y+4) (230*$v) 12 $blue
  Text (157+230*$v) $y 73 23 (Fmt $v '0.0000') 15 $white
 }
 Text 553 345 355 26 'YuNet + SFace     0.55 s' 17 $white
 Text 553 375 355 26 'SCRFD + AdaFace   1.26 s' 17 $white
 Text 553 404 355 33 'Five warm requests per pipeline.' 14 $muted
 Text 44 452 872 35 'Verification: 103 Python passes, four documented skips, 14 JavaScript passes.' 17 $cyan $true

 Slide 'AI collaboration and engineering ownership' 'DEVELOPER-DIRECTED DECISIONS' 'Sources  AI_COLLAB and developer-verified history links' "$doc/AI_COLLAB.md"
 Head 44 125 410 'Contributions'
 Bullets 44 163 402 @('The developer credits ChatGPT with research reasoning and experiment planning.','Codex implemented, investigated and verified against source and saved evidence.','The developer credits Claude with independent review of startup, rights, methodology and documentation.') 77 18
 Head 480 125 436 'Developer decisions'
 Bullets 480 163 436 @('Investigate real enrollment misses before changing detector settings.','Keep strict review and manual correction when a size heuristic chooses the wrong target.','Require independent ground truth before presenting correctness metrics.') 77 18
 Rule 418
 Text 44 435 873 56 'Published share links and seven recovered Codex sessions support the collaboration record. Earlier history and the first Codex development prompt remain incomplete.' 16 $muted

 Slide 'Live application demonstration' 'INTERVIEW WALKTHROUGH' 'Open public repository and offline acceptance evidence' $base
 Bullets 44 126 394 @('Open FaceFind locally.','Select the folder and build the gallery.','Review ambiguous references.','Upload an unseen group photograph.','Inspect known and Unknown results.','Verify with ground truth when available.','Export recognition or evaluation results.') 37 17
 Image 'group-photo-recognition' 436 123 480 248
 Image 'offline-demo' 436 385 211 99
 Text 660 382 256 96 'The developer confirmed startup and operation while disconnected. The still shows the startup state.' 15 $muted
 Text 44 437 374 30 'github.com/aksarwar8/FaceFind' 16 $cyan $true $base

 Slide 'Detailed research results' 'APPENDIX A1 / SEPARATE COHORTS' 'Source  Research comparison tables and evidence index' "$doc/RESEARCH.md"
 Text 44 113 872 22 'WIDER FACE, 500 images and 5,615 faces / custom IoU 0.50' 16 $cyan $true
 Table 44 137 @(226,111,111,103,111,211) @(
  @('Detector','Precision','Recall','F1','Mean ms','Comparison'),
  @('YuNet','0.7718','0.5945','0.6716','99.71','Shared image set'),
  @('SCRFD-10G-KPS','0.9557','0.6267','0.7570','526.14','Shared image set'),
  @('RetinaFace MobileNet','0.8221','0.3350','0.4760','106.59','Shared image set')
 ) 13 21
 Text 44 230 872 22 'LFW recognition-only / processed 578 known and 442 unknown' 16 $cyan $true
 Table 44 252 @(185,169,186,155,178) @(
  @('Recognizer','Correct known','Correct unknown','False accept','Mean embed ms'),
  @('SFace','558/578','431/442','11/442','23.21'),
  @('ArcFace','576/578','442/442','0/442','789.18'),
  @('AdaFace','576/578','432/442','10/442','55.64')
 ) 13 21
 Text 44 348 872 22 'Combined LFW / 756 known and 557 unknown with research calibration' 16 $cyan $true
 Table 44 371 @(250,220,215,188) @(
  @('Pipeline','Known correct+accepted','Unknown rejected','FPIR'),
  @('YuNet + SFace','95.11%','97.85%','2.15%'),
  @('SCRFD + AdaFace','98.28%','98.20%','1.80%')
 ) 12 22
 Text 44 446 874 50 'App timing and bundle size: YuNet + SFace 552.7 ms / 38.93 MB. SCRFD + AdaFace 1,257.4 ms / 113.03 MB.|Five warm calls each on one genuine photo. Identity correctness was unmeasured. LFW uses research settings.' 13 $muted

 Slide 'Evaluation methodology' 'APPENDIX A2 / RANK, ACCEPT, VERIFY' 'Sources  Research methods and Evaluation Studio design' "$doc/ARCHITECTURE.md#evaluation-studio"
 Head 44 120 430 'Open-set acceptance'
 Bullets 44 157 424 @('Calibrate on separate research calibration identities and freeze the operating point before evaluation.','Rank normalized embeddings by cosine similarity, the dot product of unit vectors.','AdaFace assigns a name only when the best score and the gap to second place meet their cutoffs.') 73 17
 Head 494 120 424 'Measure against independent truth'
 Bullets 494 157 422 @('Use one-to-one IoU matching at 0.50 for boxed truth, or explicit manual correspondence.','Matched faces can be correct, wrong, falsely accepted or rejected. Unmatched rows count as missed faces or false detections.','Conditional rates use valid matches. End-to-end denominators include misses and processing failures.') 73 17
 Rule 389
 Text 44 407 869 69 'Worked example: a verified known face labeled Unknown counts as a known rejection. An unknown person given a name counts as a false acceptance. A missed known face still enters the end-to-end known denominator.' 18 $white

 Slide 'Limitations and reproducibility' 'APPENDIX A3 / INTERPRETING THE EVIDENCE' 'Open setup, sources and release verification' "$doc/README.md#installation-and-first-run"
 Head 44 118 425 'Model and data limits'
 Bullets 44 156 423 @('Small, blurred or occluded faces remain difficult. Accuracy and CPU latency trade off.','A dominant-face heuristic cannot infer enrollment intent. Use strict review for uncertain references.','Five synthetic composites use approximate boxes. Four gallery identities overlap with calibration. Pretraining overlap is unknown.') 80 17
 Head 486 118 430 'Reproduce the evaluator path'
 Bullets 486 156 430 @('Use the README quick start with bundled YuNet + SFace on a local workstation.','Preferred SCRFD + AdaFace needs authorized local checkpoints. SCRFD has research restrictions and AdaFace checkpoint rights remain unresolved.','Four Python skips cover two Windows directory-symlink and two Indic shaping checks. Some earlier AI conversations were not recovered.') 80 17
 Rule 413
 Text 44 431 285 42 'Research evidence' 17 $cyan $true "$doc/docs/research/EVIDENCE.md"
 Text 338 431 282 42 'Architecture and Archify' 17 $cyan $true "$doc/ARCHITECTURE.md#archify-artifact"
 Text 644 431 270 42 'Release verification' 17 $cyan $true "$doc/docs/verification/RELEASE.md"

 $issues=@($checks | Where-Object {
  $r=$_.Shape.TextFrame.TextRange
  $r.BoundHeight -gt ($_.Height+2) -or $r.BoundWidth -gt ($_.Width+2)
 })
 if($issues.Count){
  $issues | ForEach-Object {Write-Host ("Slide {0}, {1:N1}/{2}, {3}" -f $_.Slide,$_.Shape.TextFrame.TextRange.BoundHeight,$_.Height,$_.Text)}
  throw "$($issues.Count) text box bounds need review"
 }
 $p.SaveAs((Join-Path $out 'FaceFind_Technical_Presentation.pptx'),24)
 $p.SaveAs((Join-Path $out 'FaceFind_Technical_Presentation.pdf'),32)
 if($RenderDirectory){
  [IO.Directory]::CreateDirectory($RenderDirectory)|Out-Null
  foreach($slide in $p.Slides){$slide.Export((Join-Path $RenderDirectory ('slide-{0:00}.png' -f $slide.SlideIndex)),'PNG',1600,900)}
 }
 Write-Output "Exported $($p.Slides.Count) slides"
}finally{
 $p.Close()
 $app.Quit()
}
