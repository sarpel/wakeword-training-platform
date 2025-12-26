# fast_delete_windows.py
# Windows 11: 100k+ dosyayı GUI'yi kilitlemeden daha güvenli/seri silme
# Kullanım:
#   python fast_delete_windows.py "D:\path\to\folder"
# Opsiyonel:
#   python fast_delete_windows.py "C:\Users\Sarpel\Desktop\project_1\data\npy" --keep-root          <----------- BUUUU
#   python fast_delete_windows.py "D:\path\to\folder" --keep-root   (içini boşaltır, klasörü bırakır)

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

def is_windows() -> bool:
    return os.name == "nt"

def run(cmd: list[str]) -> int:
    # CREATE_NO_WINDOW: ekstra console penceresi açmasın
    CREATE_NO_WINDOW = 0x08000000
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=CREATE_NO_WINDOW,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line.rstrip())
    return p.wait()

def robocopy_purge(target: Path) -> bool:
    """
    Robocopy ile boş bir klasörü hedefe mirror ederek (purge) hedefi hızlıca boşalt.
    Çok dosyada genelde en az GUI etkisi ve iyi performans.
    """
    empty_dir = Path(tempfile.mkdtemp(prefix="empty_robocopy_"))
    try:
        # /MIR = mirror (ekstra dosyaları siler), /R:0 /W:0 retry bekleme yok
        # /NFL /NDL listelemeyi azaltır, /NJH /NJS header/summary azaltır
        cmd = [
            "robocopy",
            str(empty_dir),
            str(target),
            "/MIR",
            "/R:0",
            "/W:0",
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
        ]
        print(">> robocopy purge başlıyor...")
        code = run(cmd)

        # Robocopy dönüş kodları bitmask'tir; 0 ve 1 genelde başarı sayılır.
        # 0: hiçbir şey kopyalanmadı/silinmedi, 1: bazı dosyalar kopyalandı (burada boş klasör, genelde 1/2/3 görülebilir)
        # 8+ hata.
        if code >= 8:
            print(f"!! robocopy hata kodu: {code}")
            return False
        return True
    finally:
        shutil.rmtree(empty_dir, ignore_errors=True)

def remove_dir_quick(target: Path) -> bool:
    """
    Klasörü komple kaldır (en hızlısı çoğu senaryoda).
    """
    cmd = ["cmd", "/c", "rmdir", "/s", "/q", str(target)]
    print(">> rmdir (tam silme) başlıyor...")
    code = run(cmd)
    return code == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Silinecek klasör yolu")
    ap.add_argument("--keep-root", action="store_true", help="Klasörü silme, sadece içini boşalt")
    args = ap.parse_args()

    if not is_windows():
        print("Bu script Windows içindir (cmd/robocopy kullanıyor).")
        sys.exit(2)

    target = Path(args.path).resolve()
    if not target.exists():
        print("Hedef bulunamadı:", target)
        sys.exit(1)
    if not target.is_dir():
        print("Hedef bir klasör olmalı:", target)
        sys.exit(1)

    # Çok temel güvenlik: kök dizinleri yanlışlıkla silmeyelim
    if str(target) in [str(Path(p).resolve()) for p in [Path("C:\\"), Path("D:\\"), Path("E:\\")]]:
        print("Güvenlik: disk kökünü silmeyi reddediyorum:", target)
        sys.exit(1)

    print("Hedef:", target)
    print("Not: Çöp kutusu kullanılmaz (kalıcı siler).")

    if args.keep_root:
        ok = robocopy_purge(target)
        if not ok:
            print("robocopy purge başarısız; alternatif yöntem deneniyor...")
            # Alternatif: içi boşaltılamadıysa, rmdir + mkdir ile kökü yeniden oluştur
            parent = target.parent
            name = target.name
            ok2 = remove_dir_quick(target)
            if ok2:
                (parent / name).mkdir(parents=True, exist_ok=True)
                print(">> Kök klasör yeniden oluşturuldu.")
                sys.exit(0)
            sys.exit(1)
        print(">> İç boşaltıldı.")
        sys.exit(0)
    else:
        # Önce direkt rmdir dene (genelde en hızlı)
        ok = remove_dir_quick(target)
        if ok:
            print(">> Klasör silindi.")
            sys.exit(0)

        print("rmdir başarısız; içi purge edilip tekrar denenecek...")
        ok2 = robocopy_purge(target)
        if ok2 and remove_dir_quick(target):
            print(">> Klasör silindi (purge + rmdir).")
            sys.exit(0)

        print("!! Silme başarısız. Yetki/lock/antivirüs engeli olabilir.")
        sys.exit(1)

if __name__ == "__main__":
    main()
