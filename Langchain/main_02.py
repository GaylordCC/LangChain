from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from dotenv import load_dotenv
import os

from docxtpl import DocxTemplate



load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

document_information = {
    'FECHA_DE_LA_CARTA' : "07 de Mayo de 2024",
    'ESTADO_ENTIDAD_UBICACION_NOTARIA' : "Barranquilla-Atlántico",
    'NOMBRE_APODERADO_LEGAL' : "Alex Char",
    'NOMBRE_ACREDITADO' : "Dissu Enrique",
    'NUMERO_DE_ESCRITURA' : "005214",
    'FECHA_DE_LA_ESCRITURA' : "07 de Febrero de 2023",
    'NOMBRE_NOTARIO' : "Eduardo Verano",
    'NUMERO_DE_LA_NOTARIA' : "33",
    'DESCRIPCION_DEL_INMUEBLE' : "Apartamento",
    'NOMBRE_COMPLETO_DEL_BANCO' : "COLPATRIA",
    'MONTO_CREDITO' : "4000",
    '- ' : ""
}

file_path = "Langchain/template03.docx"

@app.post("/generar_documento_word")
async def generar_documento_word():
    try:
        # Cargar la plantilla de Word
        doc = DocxTemplate(file_path)

        # Aplicar reemplazo en la plantilla
        context = {}
        for clave, valor in document_information.items():
            context[clave] = valor

        doc.render(context)

        # Guardar el documento generado
        generated_file_path = "generated_document.docx"
        doc.save(generated_file_path)

        # Devolver el archivo generado para su descarga
        response = FileResponse(generated_file_path)

        # Establecer cabeceras para descargar el archivo
        response.headers["Content-Disposition"] = f"attachment; filename={os.path.basename(generated_file_path)}"

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))