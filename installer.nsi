; SoundboardEZ NSIS installer (modern, machine-wide)
; Requires NSIS (Unicode build) on Windows.

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"

!define APPNAME "SoundboardEZ"
!define COMPANY "SoundboardEZ"
!define /ifndef APPVERSION "1.0.0"
!define EXE_NAME "SoundboardEZ.exe"
!define INSTALLER_NAME "SoundboardEZ-Setup.exe"
!define APPDIR "$PROGRAMFILES64\\${APPNAME}"
!define STARTMENU_FOLDER "${APPNAME}"
!define APPREG_KEY "Software\\${COMPANY}\\${APPNAME}"
!define UNINSTALL_KEY "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}"

Name "${APPNAME}"
OutFile "${INSTALLER_NAME}"
RequestExecutionLevel admin
InstallDir "${APPDIR}"
InstallDirRegKey HKLM "${APPREG_KEY}" "InstallDir"
ShowInstDetails show
ShowUnInstDetails show
SetCompressor /SOLID lzma
BrandingText "${APPNAME} Installer"

!define MUI_ABORTWARNING
!define MUI_ICON "assets\\app.ico"
!define MUI_UNICON "assets\\app.ico"
!define MUI_FINISHPAGE_RUN "$INSTDIR\\${EXE_NAME}"
!define MUI_FINISHPAGE_RUN_TEXT "Launch SoundboardEZ"
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\\README_INSTALL.txt"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "View install notes"

Var StartMenuFolder
Var VBCableSetupExitCode

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  ${If} ${RunningX64}
    SetRegView 64
  ${EndIf}
  SetShellVarContext all

  ReadRegStr $0 HKLM "${APPREG_KEY}" "InstallDir"
  ${If} $0 != ""
    StrCpy $INSTDIR $0
  ${EndIf}

  IfFileExists "$INSTDIR\\Uninstall.exe" 0 done
  MessageBox MB_ICONQUESTION|MB_YESNO \
    "${APPNAME} is already installed in:$\r$\n$INSTDIR$\r$\n$\r$\nDo you want to uninstall the existing version first?" \
    IDYES do_uninstall IDNO done

do_uninstall:
  ExecWait '"$INSTDIR\\Uninstall.exe" /S _?=$INSTDIR' $1
  StrCmp $1 0 done
  MessageBox MB_ICONEXCLAMATION "Uninstall exited with code $1. Setup will continue."

done:
FunctionEnd

Function RunVBCableInstaller
  StrCpy $VBCableSetupExitCode 0
  IfFileExists "$INSTDIR\\VBCABLE_Driver_Pack43.zip" 0 done

  CreateDirectory "$INSTDIR\\vb-cable"
  StrCpy $0 "$\"$SYSDIR\\WindowsPowerShell\\v1.0\\powershell.exe$\" -NoProfile -ExecutionPolicy Bypass -Command $\"Expand-Archive -LiteralPath '$INSTDIR\\VBCABLE_Driver_Pack43.zip' -DestinationPath '$INSTDIR\\vb-cable' -Force$\""
  ExecWait $0 $1
  ${If} $1 != 0
    MessageBox MB_ICONEXCLAMATION "VB-Cable extraction failed (code $1). Run setup manually from $INSTDIR\\vb-cable."
    Goto done
  ${EndIf}

  IfFileExists "$INSTDIR\\vb-cable\\VBCABLE_Setup_x64.exe" 0 +6
    ExecWait '"$INSTDIR\\vb-cable\\VBCABLE_Setup_x64.exe"' $VBCableSetupExitCode
    ${If} $VBCableSetupExitCode != 0
      MessageBox MB_ICONEXCLAMATION "VB-Cable x64 setup returned code $VBCableSetupExitCode. Approve driver prompts and reboot if requested."
    ${EndIf}
    Goto done

  IfFileExists "$INSTDIR\\vb-cable\\VBCABLE_Setup.exe" 0 missing
    ExecWait '"$INSTDIR\\vb-cable\\VBCABLE_Setup.exe"' $VBCableSetupExitCode
    ${If} $VBCableSetupExitCode != 0
      MessageBox MB_ICONEXCLAMATION "VB-Cable setup returned code $VBCableSetupExitCode. Approve driver prompts and reboot if requested."
    ${EndIf}
    Goto done

missing:
  MessageBox MB_ICONEXCLAMATION "VB-Cable setup executable was not found in $INSTDIR\\vb-cable."

done:
FunctionEnd

Section "Install"
  SetOutPath "$INSTDIR"

  ; Main app
  File "dist\\SoundboardEZ.exe"

  ; Optional sounds (if present)
  IfFileExists "sounds\\*.*" 0 +3
    CreateDirectory "$INSTDIR\\sounds"
    File /r "sounds\\*.*"

  ; Assets
  IfFileExists "assets\\app.ico" 0 +2
    File "/oname=$INSTDIR\\app.ico" "assets\\app.ico"

  IfFileExists "assets\\VBCABLE_Driver_Pack43.zip" 0 +2
    File "/oname=$INSTDIR\\VBCABLE_Driver_Pack43.zip" "assets\\VBCABLE_Driver_Pack43.zip"

  ; README
  IfFileExists "README_INSTALL.txt" 0 +2
    File "/oname=$INSTDIR\\README_INSTALL.txt" "README_INSTALL.txt"

  ; Uninstaller
  WriteUninstaller "$INSTDIR\\Uninstall.exe"

  ; VB-Cable install in same execution
  Call RunVBCableInstaller

  ; Shortcuts
  StrCpy $StartMenuFolder "${STARTMENU_FOLDER}"
  CreateDirectory "$SMPROGRAMS\\$StartMenuFolder"
  CreateShortCut "$SMPROGRAMS\\$StartMenuFolder\\${APPNAME}.lnk" "$INSTDIR\\${EXE_NAME}" "" "$INSTDIR\\app.ico"
  CreateShortCut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\${EXE_NAME}" "" "$INSTDIR\\app.ico"

  ; Registry entries
  WriteRegStr HKLM "${APPREG_KEY}" "InstallDir" "$INSTDIR"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "DisplayName" "${APPNAME}"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "DisplayVersion" "${APPVERSION}"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "Publisher" "${COMPANY}"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\\app.ico"
  WriteRegStr HKLM "${UNINSTALL_KEY}" "UninstallString" '"$INSTDIR\\Uninstall.exe"'
  WriteRegStr HKLM "${UNINSTALL_KEY}" "QuietUninstallString" '"$INSTDIR\\Uninstall.exe" /S'
  WriteRegDWORD HKLM "${UNINSTALL_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINSTALL_KEY}" "NoRepair" 1
SectionEnd

Function un.onInit
  ${If} ${RunningX64}
    SetRegView 64
  ${EndIf}
  SetShellVarContext all
FunctionEnd

Section "Uninstall"
  ; Remove shortcuts
  Delete "$SMPROGRAMS\\${STARTMENU_FOLDER}\\${APPNAME}.lnk"
  RMDir "$SMPROGRAMS\\${STARTMENU_FOLDER}"
  Delete "$DESKTOP\\${APPNAME}.lnk"

  ; Remove installed files (leave driver install in system)
  Delete "$INSTDIR\\Uninstall.exe"
  Delete "$INSTDIR\\${EXE_NAME}"
  Delete "$INSTDIR\\app.ico"
  Delete "$INSTDIR\\README_INSTALL.txt"
  Delete "$INSTDIR\\VBCABLE_Driver_Pack43.zip"
  RMDir /r "$INSTDIR\\sounds"
  RMDir /r "$INSTDIR\\vb-cable"
  RMDir "$INSTDIR"

  ; Remove registry
  DeleteRegKey HKLM "${UNINSTALL_KEY}"
  DeleteRegKey HKLM "${APPREG_KEY}"
SectionEnd
