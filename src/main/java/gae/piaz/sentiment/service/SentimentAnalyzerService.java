package gae.piaz.sentiment.service;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;

@Service
public class SentimentAnalyzerService {

    private static final Logger logger = LoggerFactory.getLogger(SentimentAnalyzerService.class);

    private final Predictor<String, Classifications> predictor;

    public SentimentAnalyzerService(Predictor<String, Classifications> predictor) {
        this.predictor = predictor;
    }


    /**
     * Analyzes sentiment of the provided text and returns only the sentiment score.
     *
     * @param text The text to analyze
     * @return Sentiment score between -1 (negative) and 1 (positive)
     */
    public Double analyzeSentimentSimple(String text) {
        try {
            logger.debug("Analyzing text: {}", text);
            Classifications result = predictor.predict(text);
            return calculateSentimentScore(result.items());
        } catch (Exception e) {
            logger.error("Error during sentiment analysis for text: {}", text, e);
            return 0.0; // Return neutral sentiment on error
        }
    }



    private double calculateSentimentScore(List<Classifications.Classification> classifications) {
        double positiveScore = 0.0;
        double negativeScore = 0.0;
        double neutralScore = 0.0;
        double totalWeight = 0.0;

        for (Classifications.Classification classification : classifications) {
            double weight = classification.getProbability();
            totalWeight += weight;
            
            String className = classification.getClassName().toLowerCase();
            switch (className) {
                case "positive" -> positiveScore += weight;
                case "negative" -> negativeScore += weight;
                case "neutral" -> neutralScore += weight;
                default -> logger.warn("Unknown classification: {}", className);
            }
        }

        if (totalWeight == 0.0) {
            return 0.0; // Default to neutral if no classifications
        }

        // Normalize probabilities
        double pPos = positiveScore / totalWeight;
        double pNeg = negativeScore / totalWeight;
        double pNeu = neutralScore / totalWeight;

        // Calculate sentiment score: positive - negative, dampened by neutral
        return (pPos - pNeg) * (1.0 - pNeu);
    }

}
