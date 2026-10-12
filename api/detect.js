export const config = {
  api: {
    bodyParser: false
  }
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Method not allowed."
    });
  }

  const backendUrl = process.env.MODEL_API_URL;

  if (!backendUrl) {
    return res.status(503).json({
      error: "Model backend is not configured yet."
    });
  }

  try {
    const chunks = [];

    for await (const chunk of req) {
      chunks.push(chunk);
    }

    const body = Buffer.concat(chunks);

    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": req.headers["content-type"]
      },
      body
    });

    const contentType = response.headers.get("content-type") || "";

    if (!response.ok) {
      return res.status(502).json({
        error: "The model backend returned an error."
      });
    }

    if (contentType.includes("application/json")) {
      const result = await response.json();
      return res.status(200).json(result);
    }

    return res.status(502).json({
      error: "The backend must return JSON containing the detection image and results."
    });

  } catch (error) {
    console.error("Inference request failed:", error.message);

    return res.status(502).json({
      error: "Could not connect to the model backend."
    });
  }
}
