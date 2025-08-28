package gae.piaz.sentiment.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import gae.piaz.sentiment.service.SentimentAnalyzerService;

@RestController
@RequestMapping("/api/sentiment")
@CrossOrigin
public class SentimentController {

    private static final Logger logger = LoggerFactory.getLogger(SentimentController.class);

    private final SentimentAnalyzerService sentimentAnalyzerService;

    public SentimentController(SentimentAnalyzerService sentimentAnalyzerService) {
        this.sentimentAnalyzerService = sentimentAnalyzerService;
    }

    @GetMapping("/analyze")
    public ResponseEntity<Double> analyzeSentimentSimple(@RequestParam String text) {
        try {
            logger.info("Simple sentiment analysis for text: {}", text);
            Double sentimentScore = sentimentAnalyzerService.analyzeSentimentSimple(text);
            return ResponseEntity.ok(sentimentScore);
        } catch (Exception e) {
            logger.error("Error in simple sentiment analysis", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(0.0);
        }
    }

}
