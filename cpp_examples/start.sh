#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
CONFIGS_DIR="$SCRIPT_DIR/configs"
RESOURCES_DIR="$(dirname "$SCRIPT_DIR")/resources"

R='\033[0;31m' G='\033[0;32m' Y='\033[0;33m'
C='\033[0;36m' W='\033[1;37m' D='\033[0;90m' N='\033[0m'
INVERT='\033[7m'

DEMOS=(
  "Object Detection|YOLOv8 Detection|object_detection/yolov8_demo|yolov8_demo.json"
  "Object Detection|YOLOv5 Detection|object_detection/yolov5_demo|yolov5_demo.json"
  "Object Detection|YOLOv11 Detection|object_detection/yolov11_demo|yolov11_demo.json"
  "Object Detection|PPE Detection|object_detection/ppe_detection_demo|ppe_detection_demo.json"
  "Object Detection|Mask Detection|object_detection/mask_detection_demo|mask_detection_demo.json"
  "Pose Estimation|Body Pose|pose_estimation/body_pose_demo|body_pose_demo.json"
  "Pose Estimation|Hand Landmark|pose_estimation/hand_landmark_demo|hand_landmark_demo.json"
  "Face|SCRFD Face Detection|face/scrfd_demo|scrfd_demo.json"
  "Face|Face Detection|face/face_detection_demo|face_detection_demo.json"
  "Face|Face Emotion|face/face_emotion_demo|face_emotion_demo.json"
  "Segmentation|YOLOv8 Instance Seg|segmentation/yolov8seg_demo|yolov8seg_demo.json"
  "Segmentation|YOLO26 Instance Seg|segmentation/yolo26seg_demo|yolo26seg_demo.json"
  "Segmentation|DeepLabV3+ Semantic|segmentation/deeplabv3_demo|deeplabv3_demo.json"
  "Classification|YOLO26 Classification|classification/yolo26cls_demo|yolo26cls_demo.json"
  "Classification|EfficientNet|classification/efficientnet_demo|efficientnet_demo.json"
  "Classification|MobileNetV2|classification/mobilenet_demo|mobilenet_demo.json"
  "PPU|YOLOv8 PPU annotated|ppu/yolov8_ppu_demo|yolov8_ppu_demo.json"
  "PPU|YOLOv5 PPU annotated|ppu/yolov5_ppu_demo|yolov5_ppu_demo.json"
  "PPU|YOLOv11 PPU annotated|ppu/yolov11_ppu_demo|yolov11_ppu_demo.json"
  "Async|YOLOv8 Async Pipeline|async_example/yolov8_async_demo|yolov8_async_demo.json"
  "Advanced|People Tracking|advanced/people_tracking_demo|people_tracking_demo.json"
  "Advanced|Trespassing Detection|advanced/trespassing_demo|trespassing_demo.json"
  "Advanced|Smart Traffic Counter|advanced/smart_traffic_demo|smart_traffic_demo.json"
  "Advanced|Store Queue Analysis|advanced/store_queue_demo|store_queue_demo.json"
  "Advanced|Multi-Channel 4x|advanced/multi_channel_demo|multi_channel_demo.json"
  "Advanced|Person Re-ID OSNet|advanced/osnet_reid_demo|osnet_reid_demo.json"
)

TOTAL=${#DEMOS[@]}
SEL=0
SCROLL=0
HLINES=7
FLINES=4

cfg_field() {
  local f="$CONFIGS_DIR/$1" field="$2"
  [[ -f "$f" ]] || { echo ""; return; }
  grep -o "\"${field}\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$f" \
    | head -1 | sed 's/.*: *"\(.*\)"/\1/'
}

vis_rows() {
  local h; h=$(tput lines 2>/dev/null || echo 24)
  echo $(( h - HLINES - FLINES ))
}

bin_ok() { [[ -x "$BUILD_DIR/$1" ]]; }

built_count() {
  local n=0
  for e in "${DEMOS[@]}"; do
    IFS='|' read -r _c _n b _f <<< "$e"
    bin_ok "$b" && (( n++ )) || true
  done
  echo "$n"
}

draw_header() {
  local built; built=$(built_count)
  echo -e "${W}"
  echo "  ╔══════════════════════════════════════════════════════════════╗"
  echo "  ║       DeepX CPP Examples — Interactive Demo Launcher        ║"
  printf "  ║       ${G}%2d${W} / %-2d demos built                                  ║\n" "$built" "$TOTAL"
  echo "  ╚══════════════════════════════════════════════════════════════╝"
  echo -e "${N}"
}

draw_footer() {
  echo ""
  echo -e "  ${D}arrow keys  navigate    Enter  launch    r  rebuild    q  quit${N}"
  echo -e "  ${D}${G}●${D} built   ${R}●${D} not built${N}"
}

draw_list() {
  local vis; vis=$(vis_rows)
  local last_cat="" drawn=0

  for (( i=0; i<TOTAL; i++ )); do
    IFS='|' read -r cat name bin cfg <<< "${DEMOS[$i]}"

    if [[ "$cat" != "$last_cat" ]]; then
      if (( drawn >= SCROLL && drawn < SCROLL + vis )); then
        echo -e "  ${C}${cat}${N}"
      fi
      (( drawn++ ))
      last_cat="$cat"
    fi

    if (( drawn < SCROLL || drawn >= SCROLL + vis )); then
      (( drawn++ )); continue
    fi

    local dot dc
    if bin_ok "$bin"; then dot="●"; dc="$G"; else dot="●"; dc="$R"; fi

    local sp sl
    sp=$(cfg_field "$cfg" "source_path")
    sl=$(basename "$sp")

    if (( i == SEL )); then
      printf "  ${dc}%s${N} ${INVERT}${W}  %-33s  %-22s ${N}\n" "$dot" "$name" "$sl"
    else
      printf "  ${dc}%s${N}   ${W}%-33s${N}  ${D}%-22s${N}\n" "$dot" "$name" "$sl"
    fi
    (( drawn++ ))
  done
}

redraw() {
  tput cup 0 0; tput ed
  draw_header; draw_list; draw_footer
}

sync_scroll() {
  local vis; vis=$(vis_rows)
  local idx=0 lc=""
  for (( i=0; i<=SEL; i++ )); do
    IFS='|' read -r cat _r <<< "${DEMOS[$i]}"
    if [[ "$cat" != "$lc" ]]; then (( idx++ )); lc="$cat"; fi
    (( idx++ ))
  done
  if (( idx - 1 < SCROLL )); then SCROLL=$(( idx - 1 ))
  elif (( idx > SCROLL + vis )); then SCROLL=$(( idx - vis ))
  fi
  (( SCROLL < 0 )) && SCROLL=0 || true
}

launch_demo() {
  IFS='|' read -r _cat name bin cfg <<< "${DEMOS[$SEL]}"

  local binary="$BUILD_DIR/$bin"
  local st sp title model_path
  st=$(cfg_field "$cfg" "source_type")
  sp=$(cfg_field "$cfg" "source_path")
  title=$(cfg_field "$cfg" "window_title")
  model_path=$(cfg_field "$cfg" "model_path")
  [[ -z "$title" ]] && title="$name"

  # Resolve relative paths from SCRIPT_DIR
  [[ "$sp" != /* ]] && sp="$RESOURCES_DIR/${sp#resources/}"
  [[ "$model_path" != /* ]] && model_path="$RESOURCES_DIR/${model_path#resources/}"

  tput cnorm; tput rmcup

  if ! bin_ok "$bin"; then
    echo -e "\n  ${R}[ERROR]${N} Not built: ${W}$binary${N}"
    echo -e "  Press ${Y}r${N} to rebuild.\n"
    read -rp "  Press Enter to return..." _
    tput smcup; tput civis; return
  fi

  echo -e "\n  ${W}▶  ${title}${N}"
  echo -e "  ${D}${binary}${N}"
  echo -e "  ${D}${st}  →  ${sp}${N}"
  echo -e "  ${D}Press q in the demo window to stop.${N}\n"

  cd "$SCRIPT_DIR"
  "$binary" --source "$st" --path "$sp" || true

  echo -e "\n  ${G}Done.${N}"
  read -rp "  Press Enter to return to menu..." _
  tput smcup; tput civis
}

do_rebuild() {
  tput cnorm; tput rmcup
  echo -e "\n${Y}  Rebuilding...${N}\n"
  cd "$BUILD_DIR" && make -j"$(nproc)"
  echo ""
  read -rp "  Press Enter to return..." _
  tput smcup; tput civis
}

read_key() {
  local k="" s1="" s2=""
  IFS= read -r -s -n1 k
  if [[ "$k" == $'\x1b' ]]; then
    IFS= read -r -s -n1 -t 0.15 s1 || true
    IFS= read -r -s -n1 -t 0.15 s2 || true
    k="${k}${s1}${s2}"
  fi
  if [[ -z "$k" || "$k" == $'\n' || "$k" == $'\r' ]]; then
    echo "ENTER"
    return
  fi
  printf '%s' "$k"
}

cleanup() { tput cnorm; tput rmcup; }
trap cleanup EXIT INT TERM

if [[ ! -d "$BUILD_DIR" ]]; then
  echo -e "\n${R}[ERROR]${N} Build dir not found. Run ${Y}./setup.sh${N} first.\n"
  exit 1
fi

tput smcup; tput civis
sync_scroll; redraw

while true; do
  key=$(read_key)
  case "$key" in
    $'\x1b[A'|k)
      (( SEL > 0 )) && (( SEL-- )) || true
      sync_scroll; redraw ;;
    $'\x1b[B'|j)
      (( SEL < TOTAL-1 )) && (( SEL++ )) || true
      sync_scroll; redraw ;;
    $'\x1b[5~')
      SEL=$(( SEL - $(vis_rows) )); (( SEL < 0 )) && SEL=0 || true
      sync_scroll; redraw ;;
    $'\x1b[6~')
      SEL=$(( SEL + $(vis_rows) )); (( SEL >= TOTAL )) && SEL=$(( TOTAL-1 )) || true
      sync_scroll; redraw ;;
    "ENTER"|$'\n'|$'\r'|"") launch_demo; redraw ;;
    r|R) do_rebuild; redraw ;;
    q|Q|$'\x03') break ;;
    *) redraw ;;
  esac
done
