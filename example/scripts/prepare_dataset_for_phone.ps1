# Prepara o dataset valid para transferir ao celular via USB
# Uso: .\prepare_dataset_for_phone.ps1
#
# 1. Execute este script no PC
# 2. Conecte o celular via USB
# 3. Copie a pasta "valid_para_celular" para o celular (ex: Download)
# 4. No app: toque no icone de galeria > Inferencia em Lote > Selecione as imagens

$source = "D:\Docs\SafeAI\construction site\dataset_construction_site\css-data\valid"
$dest = "c:\Users\LENOVO\yolo-flutter-app\example\valid_para_celular"

if (-not (Test-Path $source)) {
    Write-Host "Pasta nao encontrada: $source"
    Write-Host "Ajuste a variavel `$source no script conforme seu dataset."
    exit 1
}

Write-Host "Copiando dataset de: $source"
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
Copy-Item -Path $source -Destination $dest -Recurse

# Se existir valid/images, garantir que as imagens estao acessiveis
$imagesPath = Join-Path $dest "images"
if (Test-Path $imagesPath) {
    Write-Host "Imagens encontradas em: valid_para_celular\images"
} else {
    Write-Host "Imagens em: valid_para_celular (ou subpasta)"
}

Write-Host ""
Write-Host "Pronto! Proximos passos:"
Write-Host "1. Conecte o celular via USB"
Write-Host "2. Copie a pasta: $dest"
Write-Host "3. No app: icone galeria > Selecionar imagens do dataset"
Write-Host "4. Navegue ate a pasta copiada e selecione as imagens"
